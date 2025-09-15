from PIL import Image, ImageOps
from pathlib import Path
import numpy as np
import sys
import fitz

def _projection_low_var_lines(arr, axis, smooth=7, z_thresh=-0.5):
    stds = arr.std(axis=axis)
    s = (stds - stds.mean()) / (stds.std() + 1e-9)
    if smooth > 1:
        k = smooth
        pad = np.pad(s, (k//2, k-1-k//2), mode='edge')
        s = np.convolve(pad, np.ones(k)/k, mode='valid')
    low = s < (s.mean() + z_thresh * s.std())

    splits = [0]
    i, L = 0, len(low)
    while i < L:
        if low[i]:
            j = i
            while j+1 < L and low[j+1]:
                j += 1
            mid = (i + j)//2
            if 3 < mid < L-4:
                splits.append(mid)
            i = j + 1
        else:
            i += 1
    splits.append(L)
    return sorted(set(splits))

def _segments_from_splits(splits, min_size_px=40):
    segs = []
    for a, b in zip(splits[:-1], splits[1:]):
        if b - a >= int(min_size_px):
            segs.append((a, b))
    return segs

def _fallback_uniform_splits(L, k):
    step = L / k
    pts = [int(round(i*step)) for i in range(k+1)]
    return list(zip(pts[:-1], pts[1:]))

def _largest_nonwhite_bbox(pil_img, thresh=245):
    g = np.asarray(ImageOps.grayscale(pil_img))
    mask = g < thresh
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return (0, 0, pil_img.width, pil_img.height)
    return (int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1)

def process_pil_image(
    img: Image.Image,
    out_dir: Path,
    page_tag: str = "",
    title_trim_ratio=0.1,
    caption_trim_ratio=0.02,
    row_min_frac=0.12,
    col_min_frac=0.12,
    smooth=9,
    z_thresh=-0.45,
    rows_override=None,
    cols_per_row_override=None
):
    out_dir.mkdir(parents=True, exist_ok=True)

    bbox = _largest_nonwhite_bbox(img)
    img = img.crop(bbox)

    W, H = img.size
    gray = np.asarray(ImageOps.grayscale(img)).astype(np.float32)

    if rows_override is None:
        row_splits = _projection_low_var_lines(gray, axis=1, smooth=smooth, z_thresh=z_thresh)
        row_segs = _segments_from_splits(row_splits, min_size_px=H * row_min_frac)
        if not row_segs or len(row_segs) > 8:
            row_segs = _fallback_uniform_splits(H, 5)
    else:
        row_segs = _fallback_uniform_splits(H, int(rows_override))

    saved = []

    for ri, (r0, r1) in enumerate(row_segs):
        row_img = gray[r0:r1, :]

        if cols_per_row_override is not None and ri < len(cols_per_row_override):
            k = int(cols_per_row_override[ri])
            col_segs = _fallback_uniform_splits(W, k)
        else:
            col_splits = _projection_low_var_lines(row_img, axis=0, smooth=smooth, z_thresh=z_thresh)
            col_segs = _segments_from_splits(col_splits, min_size_px=W * col_min_frac)
            if not col_segs:
                guess = 4 if (W / max(1, (r1 - r0))) > 2.0 else 2
                col_segs = _fallback_uniform_splits(W, guess)

        is_title_row = (ri == 0 and len(col_segs) <= 2)

        for ci, (c0, c1) in enumerate(col_segs):
            band_h = int((r1 - r0) * (title_trim_ratio if is_title_row else caption_trim_ratio))
            panel = img.crop((c0, r0 + band_h, c1, r1))
            base = f"row{ri+1}_col{ci+1}"
            fname = f"{base}{('_' + page_tag) if page_tag else ''}.png"
            out_path = out_dir / fname
            suffix = 2
            while out_path.exists():
                out_path = out_dir / f"{base}_{suffix}{('_' + page_tag) if page_tag else ''}.png"
                suffix += 1
            panel.save(out_path)
            saved.append(str(out_path))

    return saved

def render_pdf_to_pil_list(pdf_path: str, dpi=300):
    pil_pages = []
    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc):
        mat = fitz.Matrix(dpi/72, dpi/72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        pil_pages.append((i, img))
    doc.close()
    return pil_pages

def process_path(input_path: str):
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(input_path)

    out_dir = Path(f"./crops_{p.stem}")
    outputs = []

    if p.suffix.lower() == ".pdf":
        pages = render_pdf_to_pil_list(str(p), dpi=300)
        for page_idx, pil in pages:
            page_tag = f"p{page_idx+1}"
            outputs += process_pil_image(pil, out_dir, page_tag=page_tag)
    else:
        pil = Image.open(p).convert("RGB")
        outputs += process_pil_image(pil, out_dir)

    print(f"Saved {len(outputs)} tiles -> {out_dir}")
    return outputs

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python auto_crop_panels_pdf.py <input.pdf|image>")
        sys.exit(1)
    process_path(sys.argv[1])