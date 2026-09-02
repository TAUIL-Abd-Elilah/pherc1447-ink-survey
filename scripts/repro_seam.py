"""Measure overlap-blend seam structure in an ink_detection inference output.

With patch 128 and --overlap 0.5 the sliding window steps 64 px, and the Hann
overlap-add leaves a measurable gradient excess exactly on that lattice. It is small,
but it manufactures edges on a regular grid, and edges are what a human hunting
letterforms latches onto.

Usage:  python repro_seam.py prediction.tif [--patch 128] [--overlap 0.5]

Reports mean |gradient| on seam lines vs elsewhere, over CT-supported pixels only.
A clean overlap-add would give a ratio near 1.0.
"""
import argparse

import numpy as np
import tifffile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tif")
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--overlap", type=float, default=0.5)
    a = ap.parse_args()

    stride = int(round(a.patch * (1.0 - a.overlap)))
    img = tifffile.imread(a.tif).astype(np.float32)
    if img.ndim == 3:
        img = img.max(0)
    supported = img > 0

    gh = np.abs(np.diff(img, axis=1))
    vh = supported[:, 1:] & supported[:, :-1]
    cols = np.arange(gh.shape[1])
    on_h = (cols % stride) == (stride - 1)

    gv = np.abs(np.diff(img, axis=0))
    vv = supported[1:, :] & supported[:-1, :]
    rows = np.arange(gv.shape[0])
    on_v = (rows % stride) == (stride - 1)

    h_seam = float(gh[:, on_h][vh[:, on_h]].mean())
    h_else = float(gh[:, ~on_h][vh[:, ~on_h]].mean())
    v_seam = float(gv[on_v, :][vv[on_v, :]].mean())
    v_else = float(gv[~on_v, :][vv[~on_v, :]].mean())

    print(f"file            {a.tif}")
    print(f"shape           {img.shape}, supported {supported.mean():.3f}")
    print(f"patch {a.patch}, overlap {a.overlap} -> stride {stride}")
    print(f"horizontal      seam {h_seam:.4f}   elsewhere {h_else:.4f}   "
          f"ratio {h_seam/h_else:.3f}")
    print(f"vertical        seam {v_seam:.4f}   elsewhere {v_else:.4f}   "
          f"ratio {v_seam/v_else:.3f}")


if __name__ == "__main__":
    main()
