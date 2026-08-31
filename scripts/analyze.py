"""Score an ink_9um prediction TIFF and find the best 4 cm^2 windows.

Applies the model card's label-smoothing rescale: ink_9um trains with BCE label
smoothing 0.5, so its most confident *no-ink* output sits near 0.25, not 0.
Display/threshold on (p - 0.25) / 0.5; raw stays raw for anything quantitative.
"""
import sys, json, numpy as np, tifffile

VOXEL_UM = {"PHerc1447": 8.64, "PHerc0139": 9.362}


def rescale(p):
    """Model-card display rescale. Returns values where 0 == confident no-ink."""
    return np.clip((p.astype(np.float32) - 0.25) / 0.5, 0.0, 1.0)


def window_px(voxel_um, area_cm2=4.0):
    """Side length in pixels of a square window of `area_cm2` at this voxel size."""
    side_cm = area_cm2 ** 0.5
    return int(round(side_cm * 10000.0 / voxel_um))


def best_windows(m, side, topk=5, stride=None):
    """Mean rescaled score over every `side`x`side` window, via integral image."""
    if stride is None:
        stride = max(1, side // 4)
    h, w = m.shape
    if h < side or w < side:
        return []
    ii = np.zeros((h + 1, w + 1), dtype=np.float64)
    ii[1:, 1:] = m.cumsum(0).cumsum(1)
    ys = range(0, h - side + 1, stride)
    xs = range(0, w - side + 1, stride)
    out = []
    for y in ys:
        for x in xs:
            s = ii[y + side, x + side] - ii[y, x + side] - ii[y + side, x] + ii[y, x]
            out.append((s / (side * side), y, x))
    out.sort(reverse=True)
    return out[:topk]


def main(path, scroll="PHerc1447"):
    p = tifffile.imread(path)
    if p.ndim == 3:
        p = p.max(0)
    if p.dtype == np.uint8:
        p = p.astype(np.float32) / 255.0
    elif p.dtype == np.uint16:
        p = p.astype(np.float32) / 65535.0
    m = rescale(p)
    vox = VOXEL_UM.get(scroll, 8.64)
    side = window_px(vox)
    covered = m > 0  # anything above confident-no-ink
    res = {
        "file": path,
        "shape": list(m.shape),
        "voxel_um": vox,
        "window_side_px_for_4cm2": side,
        "raw_min": round(float(p.min()), 4),
        "raw_max": round(float(p.max()), 4),
        "raw_mean": round(float(p.mean()), 4),
        "rescaled_mean": round(float(m.mean()), 4),
        "frac_above_0.25_rescaled": round(float((m > 0.25).mean()), 5),
        "frac_above_0.50_rescaled": round(float((m > 0.50).mean()), 5),
        "frac_above_0.75_rescaled": round(float((m > 0.75).mean()), 5),
        "frac_any_signal": round(float(covered.mean()), 5),
    }
    bw = best_windows(m, side)
    res["top_4cm2_windows"] = [
        {"mean_rescaled": round(float(s), 4), "y": int(y), "x": int(x)} for s, y, x in bw
    ]
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "PHerc1447")
