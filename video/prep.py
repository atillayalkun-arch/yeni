"""Ön hazırlık: temiz arka plan + eksik öğelerin (lamba, priz, etiketler, noktalı iz, gölge) katman olarak kesilmesi."""
import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from video.common import *
from scipy import ndimage as ndi

OUT = 'video/layers/'
O, P, al, rgb = load()
rng = np.random.default_rng(7)
gray = O.mean(2)

# ---------- düzlemler ve oda çizgileri (measurements.json) ----------
M = json.load(open('measurements.json'))['room_lines']
L, R, T, B = 272.5, 1105.5, 138.5, 628.0
BLX, BRX = 35.0, 1330.0   # y=767'deki alt köşe çizgisi x değerleri
TLX, TRX = 50.5, 1326.5
planes = {
    'ceiling': [(TLX, 0), (TRX, 0), (R, T), (L, T)],
    'back':    [(L, T), (R, T), (R, B), (L, B)],
    'left':    [(0, 0), (TLX, 0), (L, T), (L, B), (BLX, 767), (0, 767)],
    'right':   [(TRX, 0), (W, 0), (W, 767), (BRX, 767), (R, B), (R, T)],
    'floor':   [(L, B), (R, B), (BRX, 767), (BLX, 767)],
}
def pm(poly):
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.array(poly, np.float32).round().astype(np.int32)], 1)
    return m
pmask = {k: pm(v) for k, v in planes.items()}

# çizgiler: (x0,y0,x1,y1, kalınlık)
def ext(p0, p1, e=8):
    d = np.subtract(p1, p0); d = d / np.hypot(*d)
    return (p0[0] - d[0] * e, p0[1] - d[1] * e, p1[0] + d[0] * e, p1[1] + d[1] * e)
lines = [ext((L, T), (R, T), 0) + (2.4,), ext((L, B), (R, B), 0) + (2.6,),
         ext((L, T), (L, B), 0) + (2.4,), ext((R, T), (R, B), 0) + (2.4,),
         ext((TLX, 0), (L, T), 0) + (2.7,), ext((TRX, 0), (R, T), 0) + (2.7,),
         ext((L, B), (BLX, 767), 0) + (2.7,), ext((R, B), (BRX, 767), 0) + (2.7,)]
# çizgileri canvas dışına uzat
lines[4] = (TLX - 1.6 * 6, -6, L, T, 2.7); lines[5] = (TRX + 1.6 * 6, -6, R, T, 2.7)
lines[6] = (L, B, BLX - 1.7 * 6, 773, 2.7); lines[7] = (R, B, BRX + 1.6 * 6, 773, 2.7)
SS = 4
def draw_lines():
    big = np.zeros((H * SS, W * SS), np.uint8)
    for x0, y0, x1, y1, t in lines:
        cv2.line(big, (int(round(x0 * SS)), int(round(y0 * SS))), (int(round(x1 * SS)), int(round(y1 * SS))),
                 255, max(1, int(round(t * SS))), cv2.LINE_8)
    return cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
line_a = draw_lines()
line_zone = cv2.dilate((line_a > 0.02).astype(np.uint8), np.ones((9, 9), np.uint8))
# çizgi rengi: duvar üst çizgisinin koyu pikselleri
dk = (O.max(2) < 55) & (line_a > 0.6)
line_col = np.median(O[dk], 0) if dk.sum() > 50 else np.array([26, 22, 28.])
print('çizgi rengi', line_col, dk.sum())

# ---------- asetler ve eksik öğeler ----------
A_all = np.max([a for a in al.values()], 0)
Ad = cv2.dilate((A_all > 0.02).astype(np.uint8), np.ones((9, 9), np.uint8))
comp0 = P.copy()
for f, x, y in [ASSETS[4], ASSETS[0], ASSETS[1], ASSETS[2], ASSETS[3]]:
    comp0 = comp0 * (1 - al[f][..., None]) + rgb[f] * al[f][..., None]
D = cv2.GaussianBlur(np.abs(O - comp0).max(2), (0, 0), 1.2)
miss = ((D > 25) & (Ad == 0)).astype(np.uint8)
n, lab, st, _ = cv2.connectedComponentsWithStats(cv2.dilate(miss, np.ones((5, 5), np.uint8)))
objs = {}
names = {(562, 0): 'lamp', (306, 52): 'pill3', (842, 145): 'pill2', (1154, 229): 'switch', (1088, 439): 'pill1'}
for i in range(1, n):
    if st[i, 4] < 500: continue
    key = min(names, key=lambda k: abs(k[0] - st[i, 0]) + abs(k[1] - st[i, 1]))
    m = ((lab == i) & (miss > 0)).astype(np.uint8)
    pts = cv2.findNonZero(m)
    hull = cv2.convexHull(pts)
    hm = np.zeros((H, W), np.uint8); cv2.fillConvexPoly(hm, hull, 1)
    objs[names[key]] = hm
print('kesilen nesneler', {k: int(v.sum()) for k, v in objs.items()})

# noktalı iz: beyaz-şapka (top-hat) ile nokta tespiti
def tophat_dots(x0, y0, x1, y1, thr=14):
    g = gray.copy()
    th = g - cv2.morphologyEx(g, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    roi = np.zeros((H, W), bool); roi[y0:y1, x0:x1] = True
    m = (th > thr) & roi & (Ad == 0) & (line_zone == 0)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    keep = [i for i in range(1, n) if 4 <= st[i, 4] <= 120]
    return lab, keep, cen
labA, keepA, cenA = tophat_dots(1030, 380, 1078, 484)   # çizgi A: balon1 -> balon2 (dikey)
labB, keepB, cenB = tophat_dots(668, 180, 830, 250)     # çizgi B: balon2 -> balon3 (sol yay)
print('nokta sayısı A,B:', len(keepA), len(keepB)); print('A',sorted([(round(cenA[i][0]),round(cenA[i][1])) for i in keepA],key=lambda p:p[1])); print('B',sorted([(round(cenB[i][0]),round(cenB[i][1])) for i in keepB]))
dotmask = np.zeros((H, W), np.uint8)
for lab_, keep in ((labA, keepA), (labB, keepB)):
    for i in keep: dotmask[lab_ == i] = 1

# ---------- boşluk doldurma ----------
def fill_plane(img, valid, pmask_p):
    """Bir düzlemin temiz piksellerinden: 2. derece polinom eğilimi + çok ölçekli Gauss ağırlıklı ortalama (artık)."""
    ys, xs = np.nonzero((valid > 0) & (pmask_p > 0))
    if len(ys) > 40000:
        sel = rng.choice(len(ys), 40000, replace=False); ys, xs = ys[sel], xs[sel]
    def feats(x, y):
        x = (x - W / 2) / W; y = (y - H / 2) / H
        return np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], -1)
    Fm = feats(xs.astype(np.float32), ys.astype(np.float32))
    coef = np.linalg.lstsq(Fm, img[ys, xs], rcond=None)[0]
    gy, gx = np.mgrid[0:H, 0:W].astype(np.float32)
    trend = (feats(gx, gy) @ coef).astype(np.float32)
    res = img - trend
    v = (valid * pmask_p).astype(np.float32)
    out = None
    for s in (192, 96, 48, 24, 12, 6, 3):
        num = cv2.GaussianBlur(res * v[..., None], (0, 0), s)
        den = cv2.GaussianBlur(v, (0, 0), s)[..., None]
        est = num / np.maximum(den, 1e-4)
        w = np.clip((den - 0.05) / 0.25, 0, 1)
        out = est * w if out is None else est * w + out * (1 - w)
    return trend + out

def rebuild(mask):
    """mask: 1 = doldurulacak. Bütün düzlemleri doldurup birleştir."""
    filled = O.copy()
    hole = (mask > 0)
    valid_base = (~hole) & (line_zone == 0)
    est = np.zeros_like(O)
    for k, pmk in pmask.items():
        er = cv2.erode(pmk, np.ones((5, 5), np.uint8))
        valid = valid_base & (er > 0)
        e = fill_plane(O, valid, er)
        sel = pmk > 0
        est[sel] = e[sel]
    # çizgi bölgesi ve delikler: tahmin; dışı: orijinal
    use = hole | (line_zone > 0)
    soft = cv2.GaussianBlur(use.astype(np.float32), (0, 0), 2.0)
    soft = np.maximum(soft, use.astype(np.float32) * 0 + soft)
    soft = np.clip(soft * 1.0, 0, 1)
    grain = rng.normal(0, 1.6, O.shape).astype(np.float32)
    grain = cv2.GaussianBlur(grain, (0, 0), 0.6) * 1.4
    est = est + grain
    out = O * (1 - soft[..., None]) + est * soft[..., None]
    return np.clip(out, 0, 255), est

def dil(m, k): return cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))

mask0 = dil((A_all > 0.02), 10)
for k, v in objs.items(): mask0 |= dil(v, {'lamp': 50, 'switch': 18}.get(k, 22))
mask0 |= dil(dotmask, 15)
# bitmap'te oda çizgilerini yeniden çizeceğimiz için çizgi bölgesi de doldurulur
fill0, est0 = rebuild(mask0)

# ---------- gölge tespiti (karakter / yatak altı) ----------
ratio = (O.mean(2) + 1) / (est0.mean(2) + 1)
objall = np.zeros((H, W), np.uint8)
for v in objs.values(): objall |= v
near = dil(((al['bed.png'] > 0.3) | (al['character.png'] > 0.3)), 90)
near[:400] = 0                       # gölge yalnızca zemin/yatak-karakter çevresinde; etiket, balon ve lamba hariç
cand = ((ratio < 0.978) & (near > 0) & (Ad == 0) & (line_zone == 0) & (dotmask == 0) & (objall == 0)).astype(np.uint8)
cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
nc, lc, sc, _ = cv2.connectedComponentsWithStats(cand)
sh = np.zeros((H, W), np.uint8)
for i in range(1, nc):
    if sc[i, 4] > 120: sh[lc == i] = 1
sh = cv2.morphologyEx(dil(sh, 13), cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
print('gölge alanı', int(sh.sum()))
mask1 = mask0 | sh
clean, _ = rebuild(mask1)

# çizgileri 4x süper örnekleme ile yeniden çiz
la = line_a[..., None]
clean = clean * (1 - la) + line_col[None, None, :] * la
clean = np.clip(clean, 0, 255)
Image.fromarray(clean.round().astype(np.uint8)).save(OUT + 'bg_clean.png')

# gölge katmanı (siyah, alfa)
fill_nl = clean
sha = np.clip(1 - (O.mean(2) + 1) / (fill_nl.mean(2) + 1), 0, 0.85)
sha = sha * cv2.GaussianBlur(dil(sh, 6).astype(np.float32), (0, 0), 3)
sha[A_all > 0.6] = 0
sha[sha < 0.02] = 0
ys, xs = np.nonzero(sha > 0)
if len(ys) == 0:
    sx0, sy0, sx1, sy1 = 0, 0, 2, 2
else:
    sx0, sy0, sx1, sy1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
shl = np.zeros((sy1 - sy0, sx1 - sx0, 4), np.uint8); shl[..., 3] = (sha[sy0:sy1, sx0:sx1] * 255).round()
Image.fromarray(shl).save(OUT + 'shadow.png')

# ---------- nesne katmanları ----------
layers = {'bg': dict(file='bg_clean.png', x=0, y=0, f=0.90)}
layers['shadow'] = dict(file='shadow.png', x=int(sx0), y=int(sy0), f=0.97)
def save_rgba(name, rgbimg, alpha, x0, y0, x1, y1, f):
    a = np.clip(alpha[y0:y1, x0:x1], 0, 1)
    im = np.dstack([np.clip(rgbimg[y0:y1, x0:x1], 0, 255), a * 255]).round().astype(np.uint8)
    Image.fromarray(im).save(OUT + name + '.png')
    layers[name] = dict(file=name + '.png', x=int(x0), y=int(y0), f=f)
for k in ('lamp', 'switch', 'pill1', 'pill2', 'pill3'):
    hm = objs[k].astype(np.float32)
    a = cv2.GaussianBlur(cv2.erode(objs[k], np.ones((3, 3), np.uint8)).astype(np.float32), (0, 0), 0.9)
    if k.startswith('pill'):
        a = a * (1 - np.clip(A_all * 1.5, 0, 1))           # balonun üstünü örtme
    ys, xs = np.nonzero(objs[k])
    save_rgba(k, O, a, max(xs.min() - 3, 0), max(ys.min() - 3, 0), xs.max() + 4, ys.max() + 4, 1.07 if k.startswith('pill') else 0.92)
layers['lamp']['f'] = 0.92; layers['switch']['f'] = 0.92

# balonlar: ana gövde + küçük iz noktası (bağlı bileşenler)
for f, x, y in ASSETS[:3]:
    im = np.asarray(Image.open(f).convert('RGBA'))
    n, lab, st, _ = cv2.connectedComponentsWithStats((im[..., 3] > 12).astype(np.uint8))
    big = 1 + int(np.argmax(st[1:, 4]))
    body = im.copy(); dot = im.copy()
    body[(lab != big) & (lab > 0), 3] = 0
    dot[(lab == big) | (lab == 0), 3] = 0
    tag = f.split('_')[1]
    Image.fromarray(body).save(OUT + f'bubble{tag}.png')
    ys, xs = np.nonzero(dot[..., 3] > 0)
    dx0, dy0, dx1, dy1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    Image.fromarray(dot[dy0:dy1, dx0:dx1]).save(OUT + f'trail{tag}.png')
    layers[f'bubble{tag}'] = dict(file=f'bubble{tag}.png', x=x, y=y, f=1.07)
    layers[f'trail{tag}'] = dict(file=f'trail{tag}.png', x=int(x + dx0), y=int(y + dy0), f=1.07)
import shutil
shutil.copy('character.png', OUT + 'character.png'); layers['character'] = dict(file='character.png', x=593, y=332, f=1.0)
shutil.copy('bed.png', OUT + 'bed.png'); layers['bed'] = dict(file='bed.png', x=77, y=426, f=0.95)

# noktalı iz: toplamsal (additive) artık = orijinal - temiz arka plan; her piksel en yakın noktaya atanır
def dots_layer(name, lab_, keep, cen, order_key):
    comps = sorted(keep, key=order_key)
    seed = np.zeros((H, W), np.int32)
    for j, i in enumerate(comps): seed[lab_ == i] = j + 1
    dist, (iy, ix) = ndi.distance_transform_edt(seed == 0, return_indices=True)
    owner = seed[iy, ix]
    res = np.clip(O - clean, 0, 255)
    near_ = dist < 7
    res[~near_] = 0
    # nokta halka/parıltı: bileşen bbox ve atlas
    ys, xs = np.nonzero((owner > 0) & near_)
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    # gürültü kırp
    resm = res * (cv2.GaussianBlur(res.max(2), (0, 0), 0.8)[..., None] > 4)
    Image.fromarray(resm[y0:y1, x0:x1].round().astype(np.uint8)).save(OUT + name + '.png')
    dots = []
    for j, i in enumerate(comps):
        m = (owner == j + 1) & near_
        yy, xx = np.nonzero(m)
        dots.append(dict(x=int(xx.min() - x0), y=int(yy.min() - y0), w=int(xx.max() - xx.min() + 1), h=int(yy.max() - yy.min() + 1),
                         cx=float(cen[i][0]), cy=float(cen[i][1])))
    # sahiplik maskesi: atlas içinde her dot yalnızca kendi piksellerini göstersin
    own = np.zeros((y1 - y0, x1 - x0), np.uint8)
    own[:] = np.where(near_, owner, 0)[y0:y1, x0:x1].clip(0, 255).astype(np.uint8)
    Image.fromarray(own).save(OUT + name + '_owner.png')
    layers[name] = dict(file=name + '.png', owner=name + '_owner.png', x=int(x0), y=int(y0), f=1.0, dots=dots)
dots_layer('trailA', labA, keepA, cenA, lambda i: -cenA[i][1])   # aşağıdan yukarı
dots_layer('trailB', labB, keepB, cenB, lambda i: -cenB[i][0])   # sağdan sola
json.dump(dict(canvas=[W, H], layers=layers, line_color=line_col.tolist()), open(OUT + 'layers.json', 'w'), indent=1)

# ---------- doğrulama ----------
def over(base, f):
    im = np.asarray(Image.open(OUT + layers[f]['file']).convert('RGBA')).astype(np.float32)
    x, y = layers[f]['x'], layers[f]['y']; h, w = im.shape[:2]
    a = im[..., 3:] / 255
    base[y:y + h, x:x + w] = base[y:y + h, x:x + w] * (1 - a) + im[..., :3] * a
def addl(base, name):
    lay = layers[name]; im = np.asarray(Image.open(OUT + lay['file'])).astype(np.float32)
    x, y = lay['x'], lay['y']; h, w = im.shape[:2]
    base[y:y + h, x:x + w] += im
C = clean.copy()
over(C, 'shadow'); over(C, 'bed')
addl(C, 'trailA'); addl(C, 'trailB')
for k in ('lamp', 'switch', 'pill3', 'pill2', 'pill1', 'bubble3', 'bubble2', 'bubble1', 'trail3', 'trail2', 'trail1', 'character'):
    over(C, k)
C = np.clip(C, 0, 255)
d = np.abs(C - O)
print('BİRLEŞİK ortalama fark (tüm kare):', d.mean().round(3), ' >40 oranı:', (d.max(2) > 40).mean().round(5))
# iç bölge: asetlerin altı haricinde arka plan farkı
free = (mask1 == 0) & (line_zone == 0)
print('arka plan (maske dışı) fark, yeni vs orijinal:', np.abs(clean - O)[free].mean().round(3), ' | sağlanan arka plan vs orijinal:', np.abs(P - O)[free].mean().round(3))
Image.fromarray(C.round().astype(np.uint8)).save('/tmp/claude-0/-home-user-yeni/76adf87f-54fc-5f88-86ab-61d80db2021a/scratchpad/recomp.png')
print('dots', len(keepA), len(keepB))
