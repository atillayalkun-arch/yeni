import sys,glob
from PIL import Image, ImageDraw
d=sys.argv[1]; out=sys.argv[2]; w=int(sys.argv[3]); cols=int(sys.argv[4])
fs=sorted(glob.glob(d+'/*.jpg')); h=round(w*768/1376)
rows=(len(fs)+cols-1)//cols; S=Image.new('RGB',(cols*w,rows*h))
for i,f in enumerate(fs):
    im=Image.open(f).resize((w,h),Image.LANCZOS); dr=ImageDraw.Draw(im); n=int(f[-8:-4])
    dr.rectangle((0,0,62,14),fill=(0,0,0)); dr.text((3,1),f"{n/30:.2f}s",fill=(255,255,0))
    S.paste(im,((i%cols)*w,(i//cols)*h))
S.save(out)
