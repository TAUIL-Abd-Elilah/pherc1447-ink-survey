import zarr, fsspec, numpy as np, json, sys
B="s3://vesuvius-challenge-open-data/PHerc1447/segments"
V="surface-volumes/8.64um-1.2m-116keV-volume-20250521151220.zarr"
SEGS=["20250702235910-auto_grown_20250702235910292",
      "20250703025628-auto_grown_20250703025628283",
      "20250703034159-auto_grown_20250703034159599",
      "20251105093211-z_dbg_gen_00320"]
s3=fsspec.filesystem('s3', anon=True)
out={}
for s in SEGS:
    try:
        root=zarr.open(s3.get_mapper(f"{B}/{s}/{V}"), mode='r')
        a=root['0']
        z,y,x=a.shape
        mid=a[z//2]                      # full mid layer
        nz=float((mid>0).mean())
        vals=mid[mid>0]
        out[s]=dict(shape=[int(v) for v in a.shape],
                    nonzero_frac_midlayer=round(nz,4),
                    mean=round(float(vals.mean()),2) if vals.size else 0.0,
                    std=round(float(vals.std()),2) if vals.size else 0.0,
                    p1=int(np.percentile(vals,1)) if vals.size else 0,
                    p99=int(np.percentile(vals,99)) if vals.size else 0)
        print(s, json.dumps(out[s]), flush=True)
    except Exception as e:
        print(s, "ERR", type(e).__name__, e, flush=True)
json.dump(out, open(r"D:/Competition/Vesuvius progress prizes/_fl/gate.json","w"), indent=1)
