import numpy as np, tifffile, json, os
def resc(p):
    p=p.astype(np.float32)/255.; return np.clip((p-0.25)/0.5,0,1)
def windows(m, side, stride):
    h,w=m.shape
    if h<side or w<side: return np.array([]), None
    ii=np.zeros((h+1,w+1),np.float64); ii[1:,1:]=m.cumsum(0).cumsum(1)
    o=[]; best=(-1,0,0)
    for y in range(0,h-side+1,stride):
        for x in range(0,w-side+1,stride):
            s=(ii[y+side,x+side]-ii[y,x+side]-ii[y+side,x]+ii[y,x])/(side*side)
            o.append(float(s))
            if s>best[0]: best=(float(s),int(y),int(x))
    return np.array(o), best

JOBS=[
 ("control_PHerc0139_w043","CONTROL_0139_w043",9.362,"PHerc0139 w043 (known ink, in ink_9um training set)"),
 ("control_PHerc0139_w043_reverse","CONTROL_0139_w043_reverse",9.362,"same, reverse"),
 ("PHerc1447_seg1","1447_seg1_s42",8.64,"20250702235910-auto_grown_20250702235910292"),
 ("PHerc1447_seg1_reverse","1447_seg1_s42_reverse",8.64,"same, reverse"),
 ("PHerc1447_seg2","1447_seg2_s42",8.64,"20250703025628-auto_grown_20250703025628283"),
 ("PHerc1447_seg2_reverse","1447_seg2_s42_reverse",8.64,"same, reverse"),
 ("PHerc1447_seg2_rescaled_to_9362","1447_seg2_RESCALED9362",9.362,"seg2 XY-resampled 8.64->9.362um, scale-confound test"),
 ("PHerc1447_seg3","1447_seg3_s42",8.64,"20250703034159-auto_grown_20250703034159599"),
 ("PHerc1447_seg3_reverse","1447_seg3_s42_reverse",8.64,"same, reverse"),
 ("PHerc1447_seg4","1447_seg4_s42",8.64,"20251105093211-z_dbg_gen_00320"),
 ("PHerc1447_seg4_reverse","1447_seg4_s42_reverse",8.64,"same, reverse"),
]
out={}
for key,fn,vox,desc in JOBS:
    p=f"_fl/pred/{fn}.tif"
    if not os.path.exists(p): 
        print("missing",p); continue
    m=resc(tifffile.imread(p))
    side=int(round(2.0*10000/vox))
    w,best=windows(m, side, max(1,side//8))
    cov=float((m>0).mean())
    rec=dict(description=desc, voxel_um=vox, shape=[int(v) for v in m.shape],
             area_cm2=round(m.shape[0]*vox/10000.0 * m.shape[1]*vox/10000.0,2),
             coverage=round(cov,4),
             mean_all=round(float(m.mean()),4),
             mean_covered=round(float(m[m>0].mean()) if cov>0 else 0.0,4),
             frac_gt_050=round(float((m>0.5).mean()),5),
             frac_gt_075=round(float((m>0.75).mean()),5),
             window_side_px=side, n_windows_4cm2=int(len(w)))
    if len(w):
        rec.update(window_min=round(float(w.min()),4), window_median=round(float(np.median(w)),4),
                   window_max=round(float(w.max()),4), window_argmax_yx=[best[1],best[2]])
    out[key]=rec
    print(f"{key:34s} area={rec['area_cm2']:7.2f}cm2 n={rec['n_windows_4cm2']:3d} best={rec.get('window_max',-1):.4f}")
json.dump(out, open("_fl/results.json","w"), indent=1)
print("\nwrote _fl/results.json")
