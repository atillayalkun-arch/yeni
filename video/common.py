from PIL import Image
import numpy as np, cv2
W,H=1376,768
ASSETS=[('bubble_2_future_worry.png',777,190),('bubble_3_old_regret.png',313,89),('bubble_1_unfinished_today.png',805,462),('character.png',593,332),('bed.png',77,426)]
def load():
    O=np.asarray(Image.open('orjinal.jpeg').convert('RGB')).astype(np.float32)
    P=np.asarray(Image.open('background_room.png').convert('RGB')).astype(np.float32)
    al={};rgb={}
    for f,x,y in ASSETS:
        im=np.asarray(Image.open(f).convert('RGBA')).astype(np.float32);h,w=im.shape[:2]
        a=np.zeros((H,W),np.float32);c=np.zeros((H,W,3),np.float32)
        a[y:y+h,x:x+w]=im[...,3]/255;c[y:y+h,x:x+w]=im[...,:3]
        al[f]=a;rgb[f]=c
    return O,P,al,rgb
