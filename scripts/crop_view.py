import sys, numpy as np, tifffile
from PIL import Image
def resc(p):
    p=p.astype(np.float32)
    if p.max()>1.5: p=p/255.
    return np.clip((p-0.25)/0.5,0,1)
src,dst,y,x,s = sys.argv[1],sys.argv[2],int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5])
a=tifffile.imread(src)
if a.ndim==3: a=a.max(0)
m=resc(a)[y:y+s, x:x+s]
lo,hi=np.percentile(m,2),np.percentile(m,99.5)
img=np.clip((m-lo)/max(hi-lo,1e-6),0,1)
Image.fromarray((img*255).astype(np.uint8)).save(dst)
print(dst, m.shape, 'mean_rescaled', round(float(m.mean()),4), 'frac>0.5', round(float((m>0.5).mean()),4))
