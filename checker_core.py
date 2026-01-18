import fitz
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import List, Tuple, Dict

import torch
import open_clip


@dataclass
class Candidate:
    page: int
    bbox: Tuple[float, float, float, float]
    img: Image.Image
    embedding: np.ndarray | None


def clamp(val, lo, hi):
    return max(lo, min(hi, val))


def expand_bbox(b, margin, page_w, page_h):
    x0, y0, x1, y1 = b
    x0 = clamp(x0 - margin, 0, page_w)
    y0 = clamp(y0 - margin, 0, page_h)
    x1 = clamp(x1 + margin, 0, page_w)
    y1 = clamp(y1 + margin, 0, page_h)
    return (x0, y0, x1, y1)


def render_clip(page: fitz.Page, bbox, zoom=2.0) -> Image.Image:
    rect = fitz.Rect(*bbox)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, clip=rect, alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def get_image_blocks(page: fitz.Page) -> List[Tuple[float, float, float, float]]:
    blocks = page.get_text("dict")["blocks"]
    bboxes = []
    for bl in blocks:
        if bl.get("type") == 1 and "bbox" in bl:
            bboxes.append(tuple(bl["bbox"]))
    return bboxes


# ---- Model cache (so you don't reload CLIP every run) ----
_MODEL = None
_PREPROCESS = None
_DEVICE = None


def build_model():
    global _MODEL, _PREPROCESS, _DEVICE
    if _MODEL is not None:
        return _MODEL, _PREPROCESS, _DEVICE

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model = model.to(device)
    model.eval()

    _MODEL, _PREPROCESS, _DEVICE = model, preprocess, device
    return _MODEL, _PREPROCESS, _DEVICE


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

    # embeddings are normalized => dot = cosine sim
    for i in range(n):
        sims = embeddings[i] @ embeddings.T
        js = np.where((sims >= sim_thresh) & (np.arange(n) > i))[0]
        for j in js:
            union(i, j)

    roots = [find(i) for i in range(n)]
    root_to_id = {}
    cluster_ids = []
    next_id = 1
    for r in roots:
        if r not in root_to_id:
            root_to_id[r] = next_id
            next_id += 1
        cluster_ids.append(root_to_id[r])
    return cluster_ids


def run_checker(
    pdf_path: str,
    zoom: float = 2.0,
    margin_pts: float = 6.0,
    min_area_px: int = 20000,
    min_side_px: int = 80,
    max_aspect: float = 3.5,
    sim_thresh: float = 0.965,
    preview_max_size: Tuple[int, int] = (500, 500),
) -> List[Dict]:
    """
    Returns a list of duplicate clusters:
      {cluster_id, count, pages, preview_pil}
    No files written.
    """

    model, preprocess, device = build_model()

    doc = fitz.open(pdf_path)
    candidates: List[Candidate] = []

    for pno in range(len(doc)):
        page = doc[pno]
        page_w, page_h = page.rect.width, page.rect.height

        bboxes = get_image_blocks(page)
        if not bboxes:
            continue

        for b in bboxes:
            b2 = expand_bbox(b, margin_pts, page_w, page_h)
            img = render_clip(page, b2, zoom=zoom)

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
        return []

    embs = embed_images(model, preprocess, device, [c.img for c in candidates], batch_size=16)
    for i, c in enumerate(candidates):
        c.embedding = embs[i]

    cluster_ids = union_find_clusters(embs, sim_thresh=sim_thresh)

    # group indices by cluster
    by_cluster: Dict[int, List[int]] = {}
    for idx, cid in enumerate(cluster_ids):
        by_cluster.setdefault(cid, []).append(idx)

    # keep only duplicates
    dup_clusters = {cid: idxs for cid, idxs in by_cluster.items() if len(idxs) > 1}

    results = []
    for cid, idxs in dup_clusters.items():
        pages = sorted(set(candidates[i].page for i in idxs))  # de-dupe pages
        preview = candidates[idxs[0]].img.copy()
        preview.thumbnail(preview_max_size)
        results.append({
            "cluster_id": cid,
            "count": len(idxs),
            "pages": pages,
            "preview_pil": preview,
        })

    # Sort: most duplicates first, then earliest appearance
    results.sort(key=lambda r: (-r["count"], r["pages"][0] if r["pages"] else 10**9))
    return results
