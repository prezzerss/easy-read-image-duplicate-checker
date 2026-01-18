import os
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict

import fitz  # PyMuPDF
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import open_clip


@dataclass
class Candidate:
    page: int
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1 in PDF points
    img: Image.Image
    embedding: np.ndarray  # normalized


def clamp(val, lo, hi):
    return max(lo, min(hi, val))


def expand_bbox(b, margin, page_w, page_h):
    x0, y0, x1, y1 = b
    x0 = clamp(x0 - margin, 0, page_w)
    y0 = clamp(y0 - margin, 0, page_h)
    x1 = clamp(x1 + margin, 0, page_w)
    y1 = clamp(y1 + margin, 0, page_h)
    return (x0, y0, x1, y1)


def bbox_area(b):
    x0, y0, x1, y1 = b
    return max(0, x1 - x0) * max(0, y1 - y0)


def bbox_aspect(b):
    x0, y0, x1, y1 = b
    w = max(1e-9, x1 - x0)
    h = max(1e-9, y1 - y0)
    return w / h


def render_clip(page: fitz.Page, bbox, zoom=2.0) -> Image.Image:
    # Render just that rectangle at higher resolution
    rect = fitz.Rect(*bbox)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, clip=rect, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img


def get_image_blocks(page: fitz.Page) -> List[Tuple[float, float, float, float]]:
    """
    Uses PyMuPDF's layout analysis to find image blocks with bounding boxes.
    This is WAY better than raw embedded-image extraction for page-position info.
    """
    blocks = page.get_text("dict")["blocks"]
    bboxes = []
    for bl in blocks:
        # In PyMuPDF "dict" output: type 1 blocks are images
        if bl.get("type") == 1 and "bbox" in bl:
            bboxes.append(tuple(bl["bbox"]))
    return bboxes


def build_model(device: str = None):
    device = device or ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model = model.to(device)
    model.eval()
    return model, preprocess, device


@torch.no_grad()
def embed_images(model, preprocess, device, pil_images: List[Image.Image], batch_size=16) -> np.ndarray:
    embs = []
    for i in range(0, len(pil_images), batch_size):
        batch = pil_images[i:i+batch_size]
        tens = torch.stack([preprocess(im) for im in batch]).to(device)
        feats = model.encode_image(tens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        embs.append(feats.cpu().numpy())
    return np.vstack(embs)


def union_find_clusters(embeddings: np.ndarray, sim_thresh: float) -> List[int]:
    """
    Simple clustering: connect items if cosine similarity >= threshold.
    Works well up to a few thousand candidates. (Most PDFs will be far less.)
    """
    n = embeddings.shape[0]
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # cosine similarity matrix chunked
    # embeddings are already normalized => dot product = cosine sim
    for i in tqdm(range(n), desc="Clustering"):
        sims = embeddings[i] @ embeddings.T  # (n,)
        # Only check j > i to avoid double work
        candidates = np.where((sims >= sim_thresh) & (np.arange(n) > i))[0]
        for j in candidates:
            union(i, j)

    # compress
    roots = [find(i) for i in range(n)]
    # map root -> cluster id
    root_to_id = {}
    cluster_ids = []
    next_id = 1
    for r in roots:
        if r not in root_to_id:
            root_to_id[r] = next_id
            next_id += 1
        cluster_ids.append(root_to_id[r])
    return cluster_ids


def save_cluster_previews(out_dir: str, candidates: List[Candidate], cluster_ids: List[int], max_per_cluster=1):
    os.makedirs(out_dir, exist_ok=True)
    by_cluster: Dict[int, List[int]] = {}
    for idx, cid in enumerate(cluster_ids):
        by_cluster.setdefault(cid, []).append(idx)

    preview_paths = {}
    for cid, idxs in by_cluster.items():
        # pick first image(s) as previews
        for k, idx in enumerate(idxs[:max_per_cluster]):
            path = os.path.join(out_dir, f"cluster_{cid:03d}_{k+1}.png")
            candidates[idx].img.save(path)
            preview_paths.setdefault(cid, []).append(path)
    return preview_paths


def main(
    pdf_path: str,
    out_csv: str = "report.csv",
    previews_dir: str = "previews",
    zoom: float = 2.0,
    margin_pts: float = 6.0,
    min_area_px: int = 20_000,
    min_side_px: int = 80,
    max_aspect: float = 3.5,
    sim_thresh: float = 0.965,
):
    """
    Key knobs:
    - sim_thresh: higher => fewer false positives. Start 0.965. If it misses real duplicates try 0.955.
    - min_area_px/min_side_px: raise these to ignore icons/templates.
    - margin_pts: expands bbox so small overlay icons (next to the image) can be included.
    """

    model, preprocess, device = build_model()

    doc = fitz.open(pdf_path)
    candidates: List[Candidate] = []

    print(f"Reading PDF: {pdf_path}")
    for pno in tqdm(range(len(doc)), desc="Pages"):
        page = doc[pno]
        page_w, page_h = page.rect.width, page.rect.height

        bboxes = get_image_blocks(page)

        # If a PDF is "flattened", bboxes can be empty.
        # In that case you *can* fall back to scanning, but we keep MVP simple:
        if not bboxes:
            continue

        for b in bboxes:
            b2 = expand_bbox(b, margin_pts, page_w, page_h)
            # render the clipped region
            img = render_clip(page, b2, zoom=zoom)

            # filter small stuff (template icons etc.)
            w, h = img.size
            if w < min_side_px or h < min_side_px:
                continue
            if (w * h) < min_area_px:
                continue

            aspect = w / max(1, h)
            if aspect > max_aspect or (1/aspect) > max_aspect:
                continue

            candidates.append(Candidate(page=pno + 1, bbox=b2, img=img, embedding=None))

    if not candidates:
        print("No image blocks found. (This PDF might be flattened or has no detectable image blocks.)")
        print("If you want, I’ll add a fallback that scans rendered pages for picture regions.")
        return

    # Embed
    print(f"Embedding {len(candidates)} candidate regions on device: {device}")
    embs = embed_images(model, preprocess, device, [c.img for c in candidates], batch_size=16)
    # Already normalized
    for i, c in enumerate(candidates):
        c.embedding = embs[i]

    # Cluster
    cluster_ids = union_find_clusters(embs, sim_thresh=sim_thresh)

    # Build report rows
    df = pd.DataFrame([{
        "cluster_id": cluster_ids[i],
        "page": candidates[i].page,
        "bbox": candidates[i].bbox,
        "width_px": candidates[i].img.size[0],
        "height_px": candidates[i].img.size[1],
    } for i in range(len(candidates))])

    # Keep only clusters with duplicates
    counts = df["cluster_id"].value_counts()
    dup_clusters = counts[counts > 1].index.tolist()
    df_dup = df[df["cluster_id"].isin(dup_clusters)].copy()

    # Save previews for duplicates
    preview_paths = save_cluster_previews(previews_dir, candidates, cluster_ids, max_per_cluster=1)

    # Summarise to a friendly CSV (one row per cluster)
    summary_rows = []
    for cid in sorted(dup_clusters):
        pages = sorted(df_dup[df_dup["cluster_id"] == cid]["page"].tolist())
        summary_rows.append({
            "cluster_id": cid,
            "count": len(pages),
            "pages": " ".join(map(str, pages)),
            "preview": (preview_paths.get(cid, [""])[0] if cid in preview_paths else "")
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_csv, index=False)

    print(f"Done ✅")
    print(f"Wrote: {out_csv}")
    print(f"Previews in: {previews_dir}/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AI duplicate image checker for PDFs (CLIP embeddings).")
    parser.add_argument("pdf", help="Path to input PDF")
    parser.add_argument("--out_csv", default="report.csv")
    parser.add_argument("--previews_dir", default="previews")

    # Tuning knobs (you will tweak these)
    parser.add_argument("--sim", type=float, default=0.965, help="Cosine similarity threshold (higher = stricter)")
    parser.add_argument("--min_side", type=int, default=80, help="Min width/height in pixels to keep a region")
    parser.add_argument("--min_area", type=int, default=20000, help="Min pixel area to keep a region")
    parser.add_argument("--max_aspect", type=float, default=3.5, help="Max aspect ratio (filter out very wide/tall)")
    parser.add_argument("--margin_pts", type=float, default=6.0, help="BBox expand margin in PDF points")
    parser.add_argument("--zoom", type=float, default=2.0, help="Render zoom (2.0 ~ good default)")

    args = parser.parse_args()

    main(
        pdf_path=args.pdf,
        out_csv=args.out_csv,
        previews_dir=args.previews_dir,
        sim_thresh=args.sim,
        min_side_px=args.min_side,
        min_area_px=args.min_area,
        max_aspect=args.max_aspect,
        margin_pts=args.margin_pts,
        zoom=args.zoom,
    )
