# The D.I.G - Duplicate Image Getter

A desktop tool that detects repeated images across PDF and Pages documents using AI image embeddings.

Built to support **Easy Read Online’s amends workflow**, where the same image may appear multiple times in different contexts and needs reviewing.

---

## Features

- Detects visually similar images using AI (CLIP embeddings)
- Sorts results by **most duplicated images first**
- Shows **page numbers** for each duplicate group
- Visual preview of detected images
- Supports:
  - PDF documents
  - Apple Pages documents (auto-exported to PDF)
- Accessible, clean UI designed for internal use
- One-click launch (no terminal required)

---

## How it works

1. Extracts image regions from each page
2. Generates AI embeddings for each image
3. Clusters visually similar images
4. Displays duplicate groups ordered by impact

---

## Tech stack

- Python
- Tkinter (desktop UI)
- PyMuPDF (PDF parsing)
- OpenCLIP + PyTorch (image embeddings)
- macOS Automator launcher

---

## Why this exists

In Easy Read documents, images are often reused intentionally — but during **amends**, editors need to quickly locate *where* and *how often* an image appears.

This tool reduces manual checking time and increases confidence during review.

---

## Status

- Actively developed
- Built as part of a degree apprenticeship project
- Used internally for testing and workflow improvement

---

## Author

**Presley Dobson**  
Junior Digital Accessibility Specialist  
Easy Read Online Ltd
