import sys, numpy as np, tifffile
from PIL import Image
def rescale(p):
    if p.dtype==np.uint8: p=p.astype(np.float32)/255.
    elif p.dtype==np.uint16: p=p.astype(np.float32)/65535.
    else: p=p.astype(np.float32)
    return np.clip((p-0.25)/0.5,0,1)
src, dst = sys.argv[1], sys.argv[2]
maxw = int(sys.argv[3]) if len(sys.argv)>3 else 1600
a=tifffile.imread(src)
if a.ndim==3: a=a.max(0)
m=rescale(a)
# percentile stretch so faint strokes are visible without inventing them
lo,hi=np.percentile(m,1),np.percentile(m,99.5)
if hi<=lo: hi=lo+1e-6
img=np.clip((m-lo)/(hi-lo),0,1)
img=(img*255).astype(np.uint8)
im=Image.fromarray(img)
if im.width>maxw:
    im=im.resize((maxw,int(im.height*maxw/im.width)), Image.LANCZOS)
im.save(dst)
print(dst, im.size, "raw_range", round(float(a.min()),4), round(float(a.max()),4))
