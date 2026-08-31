"""Resample a surface-volume zarr in XY to a different effective voxel size.

Closes a confound in the PHerc1447 read: ink_9um was trained at ~9.362-9.6 um isotropic,
but PHerc1447 renders are 8.64 um, an ~8% finer scale. If that mismatch alone suppresses
the model's response, a negative on PHerc1447 would be an artifact rather than a result.

Resampling 8.64 -> 9.362 um means shrinking XY by 8.64/9.362 = 0.9229.
"""
import sys, os, shutil
import numpy as np
import zarr
from PIL import Image

src, dst, src_um, dst_um = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
f = src_um / dst_um

a = zarr.open_group(src, mode="r")["0"]
z, y, x = a.shape
ny, nx = int(round(y * f)), int(round(x * f))
print(f"{src_um} um -> {dst_um} um   factor {f:.4f}   {y}x{x} -> {ny}x{nx}", flush=True)

if os.path.isdir(dst):
    shutil.rmtree(dst)
g = zarr.open_group(dst, mode="w", zarr_format=2)
d = g.create_array("0", shape=(z, ny, nx), chunks=(z, 128, 128), dtype="u1")

for k in range(z):
    sl = np.asarray(a[k])
    # bilinear in XY only; depth is the model's channel axis and must not be touched
    d[k] = np.asarray(Image.fromarray(sl).resize((nx, ny), Image.BILINEAR))
    if k % 10 == 0:
        print(f"  layer {k}/{z}", flush=True)

src_attrs = dict(zarr.open_group(src, mode="r").attrs)
for kk, vv in src_attrs.items():
    g.attrs[kk] = vv
g.attrs["canvas_size"] = [int(nx), int(ny)]
g.attrs["resampled_from_um"] = src_um
g.attrs["resampled_to_um"] = dst_um
print("wrote", dst, flush=True)
