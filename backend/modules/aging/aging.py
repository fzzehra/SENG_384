import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _lm(landmarks, idx):
    p = landmarks[idx]
    return (int(round(p[0])), int(round(p[1])))

def _dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))

def _perlin(h, w, scale=4.0, seed=42):
    rng = np.random.default_rng(seed)
    noise = np.zeros((h, w), dtype=np.float32)
    amp, freq = 1.0, 1.0
    for _ in range(4):
        sh = max(1, int(h / (scale * freq)))
        sw = max(1, int(w / (scale * freq)))
        small = rng.random((sh, sw)).astype(np.float32) * 2 - 1
        noise += cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC) * amp
        amp *= 0.5; freq *= 2.0
    mn, mx = noise.min(), noise.max()
    return (noise - mn) / (mx - mn + 1e-6)

# ---------------------------------------------------------------------------
# Intensity calibration
# FIX: t^0.65 → t^0.40 — düşük değerlerde çok daha fazla etki
# ---------------------------------------------------------------------------
def _remap_intensity(raw):
    t = float(np.clip(raw, 0.0, 1.0))
    return t ** 0.40          # eskiden 0.65 → slider %30'da bile belirgin olsun

def _scale(intensity, lo, hi):
    return lo + intensity * (hi - lo)

# ---------------------------------------------------------------------------
# Face mask
# ---------------------------------------------------------------------------
def _build_face_mask(h, w, landmarks):
    mask = np.zeros((h, w), dtype=np.uint8)
    if landmarks is None:
        return mask
    contour = [10,338,297,332,284,251,389,356,454,323,361,288,
               397,365,379,378,400,377,152,148,176,149,150,136,
               172,58,132,93,234,127,162,21,54,103,67,109]
    pts = np.array([_lm(landmarks, i) for i in contour])
    cv2.fillPoly(mask, [pts], 255)
    return mask

# ---------------------------------------------------------------------------
# Yanak maskesi
# ---------------------------------------------------------------------------
def _build_cheek_mask(h, w, landmarks):
    mask  = np.zeros((h, w), dtype=np.float32)
    top   = _lm(landmarks, 10);  chin  = _lm(landmarks, 152)
    left  = _lm(landmarks, 234); right = _lm(landmarks, 454)
    face_w = _dist(left, right);  face_h = _dist(top, chin)
    cl = (int(left[0]  + face_w*0.12), int(top[1] + face_h*0.42))
    cr = (int(right[0] - face_w*0.12), int(top[1] + face_h*0.42))
    cv2.ellipse(mask, cl, (int(face_w*0.30), int(face_h*0.28)), 0, 0, 360, 1.0, -1)
    cv2.ellipse(mask, cr, (int(face_w*0.30), int(face_h*0.28)), 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (41, 41), 0)
    return np.clip(mask, 0, 1)

# ---------------------------------------------------------------------------
# Boyun maskesi — daha geniş bant
# ---------------------------------------------------------------------------
def _build_neck_mask(h, w, landmarks):
    mask   = np.zeros((h, w), dtype=np.float32)
    chin   = _lm(landmarks, 152); top   = _lm(landmarks, 10)
    left   = _lm(landmarks, 234); right = _lm(landmarks, 454)
    face_w = _dist(left, right);  face_h = _dist(top, chin)
    cx     = (left[0] + right[0]) // 2
    # Daha geniş ve daha uzun bant (0.38→0.50 genişlik, 0.20→0.30 yükseklik)
    neck_y = int(chin[1] + face_h * 0.06)
    cv2.ellipse(mask, (cx, neck_y),
                (int(face_w * 0.50), int(face_h * 0.30)),
                0, 0, 360, 1.0, -1)
    mask[:int(chin[1] - face_h * 0.01), :] = 0
    mask = cv2.GaussianBlur(mask, (41, 41), 0)
    return np.clip(mask, 0, 1)

# ---------------------------------------------------------------------------
# Wrinkle height-field
# ---------------------------------------------------------------------------
def _make_wrinkle_heightfield(h, w, landmarks, face_scale, seed=42):
    rng   = np.random.default_rng(seed)
    hmap  = np.zeros((h, w), dtype=np.float32)
    if landmarks is None:
        return hmap

    S      = face_scale / 250.0
    top    = _lm(landmarks, 10);   bottom = _lm(landmarks, 152)
    left   = _lm(landmarks, 234);  right  = _lm(landmarks, 454)
    face_w = _dist(left, right);   face_h = _dist(top, bottom)

    n_c = _perlin(h, w, scale=1.8, seed=seed)
    n_f = _perlin(h, w, scale=0.5, seed=seed+7)
    n_m = _perlin(h, w, scale=0.2, seed=seed+13)

    def _ridge(p1, p2, half_w, height, jitter=0.0):
        L = max(1, int(_dist(p1, p2)))
        t = np.linspace(0, 1, L + 1)
        env = np.where(t < 0.15, t / 0.15,
              np.where(t > 0.85, (1-t)/0.15, 1.0)) ** 1.8
        fade = env * height
        dx = (p2[0]-p1[0])/L;  dy = (p2[1]-p1[1])/L
        nx, ny = -dy, dx
        f1 = rng.uniform(1.5, 3.5); f2 = rng.uniform(4.5, 8.0)
        wave = (np.sin(t*np.pi*f1)*0.6 + np.sin(t*np.pi*f2)*0.4) * jitter
        # FIX: sigma daha geniş (1.6→1.1) → gaussian daha yayvan, keskin spike yok
        sigma   = max(0.8, half_w / 1.1)
        r_range = np.arange(-int(half_w*4), int(half_w*4)+1)
        gauss   = np.exp(-(r_range**2) / (2*sigma**2))
        for bx, by, f, wv in zip(
            p1[0] + (p2[0]-p1[0])*t,
            p1[1] + (p2[1]-p1[1])*t,
            fade, wave
        ):
            cx = bx + nx*wv;  cy = by + ny*wv
            px = np.round(cx + nx*r_range).astype(int)
            py = np.round(cy + ny*r_range).astype(int)
            ok = (px>=0)&(px<w)&(py>=0)&(py<h)
            py_v=py[ok]; px_v=px[ok]
            tex = n_c[py_v,px_v]*0.45 + n_f[py_v,px_v]*0.35 + n_m[py_v,px_v]*0.20
            # FIX: brk eğrisi yumuşatıldı — threshold kaldırıldı, smooth S-curve
            brk = (np.clip(tex, 0, 1) ** 1.2)
            np.add.at(hmap, (py_v, px_v), gauss[ok]*f*brk)

    w_px = max(0.8, 1.2*S)

    # ── 1. ALIN ──────────────────────────────────────────────────────────────
    n_f_lines = rng.integers(6, 9)
    for i in range(n_f_lines):
        t_pos  = (i+1)/(n_f_lines+1)
        cy_mid = int(top[1] + t_pos * face_h * 0.22)
        p1 = (int(left[0]+face_w*0.15),  cy_mid + rng.integers(-6,6))
        p2 = (int(right[0]-face_w*0.15), cy_mid + rng.integers(-6,6))
        _ridge(p1, p2, w_px*1.4, rng.uniform(0.38, 0.58), jitter=face_w*0.008)

    # Glabellar
    brow_l = _lm(landmarks, 70);  brow_r = _lm(landmarks, 300)
    for side, brow in [(-1, brow_l), (1, brow_r)]:
        bx, by = brow
        for k in range(2):
            off = int(face_w * 0.012 * k)
            p1  = (bx + side*off, by)
            p2  = (bx + side*off + rng.integers(-3,3), int(by + face_h*0.055))
            _ridge(p1, p2, w_px*1.0, rng.uniform(0.45, 0.65), jitter=face_w*0.003)

    # ── 2. GÖZ ÇEVRESİ ───────────────────────────────────────────────────────
    for eye_idx, sign in [(33,-1),(263,1)]:
        ex, ey = _lm(landmarks, eye_idx)
        for k in range(7):
            angle  = np.radians(-55 + k*20)
            length = face_w * rng.uniform(0.10, 0.17)
            _ridge((ex, ey),
                   (int(ex + sign*np.cos(angle)*length), int(ey + np.sin(angle)*length)),
                   w_px*1.0, rng.uniform(0.65, 0.92), jitter=face_w*0.006)
        _ridge((int(ex-sign*face_w*0.11), int(ey-face_h*0.008)),
               (int(ex+sign*face_w*0.11), int(ey-face_h*0.008)),
               w_px*0.9, rng.uniform(0.35, 0.52), jitter=face_w*0.004)
        for k in range(3):
            oy = face_h*(0.028 + k*0.016)
            _ridge((int(ex-sign*face_w*0.10), int(ey+oy)),
                   (int(ex+sign*face_w*0.10), int(ey+oy+face_h*0.010)),
                   w_px*0.80, rng.uniform(0.40, 0.60), jitter=face_w*0.004)
        bc = (ex, int(ey+face_h*0.058))
        for _ in range(8):
            ang = rng.uniform(0, np.pi)
            l   = face_w*rng.uniform(0.04, 0.10)
            _ridge(bc, (int(bc[0]+np.cos(ang)*l), int(bc[1]+np.sin(ang)*l*0.4)),
                   w_px*0.70, rng.uniform(0.20, 0.34), jitter=face_w*0.003)

    # ── 3. YANAK KIRIŞIKLIKLARI ───────────────────────────────────────────────
    for side, cheek_top_idx, cheek_bot_idx in [(-1, 117, 123), (1, 346, 352)]:
        ct   = _lm(landmarks, cheek_top_idx)
        cb   = _lm(landmarks, cheek_bot_idx)
        cx_c = ct[0];  cy_c = (ct[1] + cb[1]) // 2
        n_ch = rng.integers(4, 7)
        for k in range(n_ch):
            t_k     = (k+1)/(n_ch+1)
            start_x = int(cx_c + side * face_w * rng.uniform(0.04, 0.14))
            start_y = int(cy_c - face_h * 0.04 + t_k * face_h * 0.14)
            end_x   = int(start_x - side * face_w * rng.uniform(0.07, 0.14))
            end_y   = int(start_y + face_h * rng.uniform(0.04, 0.07))
            _ridge((start_x, start_y), (end_x, end_y),
                   w_px*1.0, rng.uniform(0.35, 0.55), jitter=face_w*0.006)

    # ── 4. NASOLABİAL — sadece heightfield'e hafif, geniş iz ────────────────
    # Yara izi FIX: ridge tabanlı keskin shading KALDIRILDI.
    # Yerine çok geniş (half_w*5.0) ve çok düşük yükseklikli (0.18-0.28) iz.
    # Gerçek fold efekti artık _apply_nasolabial_fold() ile warp tabanlı yapılıyor.
    nose_l  = _lm(landmarks, 102); nose_r  = _lm(landmarks, 331)
    mouth_l = _lm(landmarks, 61);  mouth_r = _lm(landmarks, 291)
    for np_, mp_, side in [(nose_l, mouth_l, -1), (nose_r, mouth_r, 1)]:
        p1 = (int(np_[0] + side * face_w * 0.010), int(np_[1] - face_h * 0.008))
        p2 = (int(mp_[0] + side * face_w * 0.012), int(mp_[1] + face_h * 0.006))
        # Çok geniş + çok düşük → sadece doku, keskin çizgi yok
        _ridge(p1, p2, w_px*5.0, rng.uniform(0.18, 0.28), jitter=face_w*0.004)

    # ── 5. MARİONETTE — daha geniş ve yumuşak ───────────────────────────────
    chin_pt = _lm(landmarks, 152)
    for m_idx, side in [(61, -1), (291, 1)]:
        mouth = _lm(landmarks, m_idx)
        p2    = (mouth[0] + int(side * face_w * 0.04) + rng.integers(-3, 3),
                 int(mouth[1] + (chin_pt[1] - mouth[1]) * 0.82))
        # half_w artırıldı (1.7→2.5), height düşürüldü (0.68-0.88→0.45-0.60)
        _ridge(mouth, p2, w_px*3.8, rng.uniform(0.22, 0.35), jitter=face_w*0.006)
        p1s = (mouth[0] + int(side*face_w*0.025), mouth[1])
        p2s = (p2[0]    + int(side*face_w*0.025), p2[1])
        _ridge(p1s, p2s, w_px*1.5, rng.uniform(0.25, 0.38), jitter=face_w*0.005)

    # ── 6. BOYUN KIRIŞIKLIKLARI — daha derin ve geniş ────────────────────────
    chin_center = _lm(landmarks, 152)
    neck_top_y  = int(chin_center[1] + face_h * 0.06)
    neck_w_half = int(face_w * 0.42)   # eskiden 0.32 → daha geniş

    n_neck = rng.integers(1, 3)        # eskiden 2-3 → daha fazla çizgi
    for i in range(n_neck):
        ny_off = int(face_h * 0.055 * i)
        p1 = (chin_center[0] - neck_w_half + rng.integers(-6, 6),
              neck_top_y + ny_off + rng.integers(-5, 5))
        p2 = (chin_center[0] + neck_w_half + rng.integers(-6, 6),
              neck_top_y + ny_off + rng.integers(-5, 5))
        # Daha derin: 0.28-0.45 → 0.42-0.62
        _ridge(p1, p2, w_px*1.5, rng.uniform(0.18, 0.30), jitter=face_w*0.008)

    # ── Post-blur: tüm ridge'leri yumuşat — kalem izleri dağılır ────────────
    hmap = cv2.GaussianBlur(hmap, (11, 11), 3.0)
 
    # ── Normalize ─────────────────────────────────────────────────────────────
    pk = hmap.max()
    if pk > 1e-6:
        hmap = hmap / pk * 0.75
    return hmap

# ---------------------------------------------------------------------------
# Nasolabial fold — warp tabanlı (yara izi yok, doğal cilt kıvrımı)
# Normal-map ridge shadingı KULLANMAZ; sadece hafif doku koyulaştırması.
# ---------------------------------------------------------------------------
def _apply_nasolabial_fold(image, landmarks, t, face_w, face_h):
    """
    Nasolabial fold: pixel-pixel elips loop'u KALDIRILDI (ağız kararması yapıyordu).
    Yerine: çizgi boyunca tek seferlik dar bir polyline maskesi çizilir,
    ardından çok geniş GaussianBlur ile tamamen diffuse edilir.
    Highlight YOK — sadece çok hafif gölge bandı.
    """
    h, w    = image.shape[:2]
    nose_l  = _lm(landmarks, 102); nose_r  = _lm(landmarks, 331)
    mouth_l = _lm(landmarks, 61);  mouth_r = _lm(landmarks, 291)

    shadow_fold = np.zeros((h, w), dtype=np.float32)

    for (nx_, ny_), (mx_, my_) in [(nose_l, mouth_l), (nose_r, mouth_r)]:
        # Polyline'ı tek cv2.line çağrısıyla çiz (loop yok → birikim yok)
        thickness = max(1, int(face_w * 0.06))
        cv2.line(shadow_fold,
                 (nx_, ny_), (mx_, my_),
                 1.0, thickness)

    # Çok geniş blur → sert kenar tamamen kaybolur, sadece hafif gradyan bant kalır
    shadow_fold = cv2.GaussianBlur(shadow_fold, (0, 0), sigmaX=face_w*0.10)
    shadow_fold = np.clip(shadow_fold, 0, 1)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    # Sadece çok hafif karartma — max 7L birimi (eskiden 10, ağız kararmasına yol açıyordu)
    lab[:,:,0] -= shadow_fold * _scale(t, 0.0, 7.0)
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

# ---------------------------------------------------------------------------
# Wrinkle application
# ---------------------------------------------------------------------------
def _apply_wrinkles(image, face_mask, intensity_raw, landmarks):
    t = _remap_intensity(intensity_raw)

    h, w       = image.shape[:2]
    face_scale = _dist(_lm(landmarks, 234), _lm(landmarks, 454))

    face_soft  = cv2.GaussianBlur(face_mask.astype(np.float32)/255.0, (31,31), 0)
    cheek_mask = _build_cheek_mask(h, w, landmarks)
    neck_mask  = _build_neck_mask(h, w, landmarks)

    H_full = _make_wrinkle_heightfield(h, w, landmarks, face_scale)

    # Yüz içi + yanak kenarları + boyun — hepsi ayrı kanalda toplanır
    face_combined = np.maximum(face_soft, cheek_mask * 0.90)
    H = H_full * face_combined + H_full * neck_mask * 1.00   # boyun tam kuvvette

    gz_val = _scale(t, 0.65, 0.18)   # FIX: daha yüksek gz → normallar daha düz → gölge yumuşak
    gx = cv2.Sobel(H, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(H, cv2.CV_32F, 0, 1, ksize=5)
    gz = np.ones_like(H) * gz_val
    nlen = np.sqrt(gx**2 + gy**2 + gz**2) + 1e-6
    nx_ = gx/nlen;  ny_ = gy/nlen;  nz_ = gz/nlen

    lx, ly, lz = -0.4, -0.6, 0.7
    shading = nx_*lx + ny_*ly + nz_*lz

    # FIX: presence eşiği düşürüldü → az intensity'de bile kırışıklar görünür
    presence_thresh = _scale(t, 0.12, 0.02)
    presence = np.clip((H - presence_thresh) * 8.0, 0, 1)

    shadow_map    = cv2.GaussianBlur(np.clip(-shading, 0, 1) * presence, (13,13), 3.0)
    highlight_map = cv2.GaussianBlur(np.clip( shading, 0, 1) * presence, (13,13), 3.0)

    # Nasolabial bölgede shadow/highlight tamamen bastır (warp fold alır görevi)
    top    = _lm(landmarks, 10);  bottom = _lm(landmarks, 152)
    left   = _lm(landmarks, 234); right  = _lm(landmarks, 454)
    face_w = _dist(left, right);  face_h = _dist(top, bottom)
    nose_l  = _lm(landmarks, 102); nose_r  = _lm(landmarks, 331)
    mouth_l = _lm(landmarks, 61);  mouth_r = _lm(landmarks, 291)

    naso_suppress = np.zeros((h, w), dtype=np.float32)
    for np_, mp_ in [(nose_l, mouth_l), (nose_r, mouth_r)]:
        cx_n = (np_[0] + mp_[0]) // 2
        cy_n = (np_[1] + mp_[1]) // 2
        cv2.ellipse(naso_suppress, (cx_n, cy_n),
                    (int(face_w*0.12), int(face_h*0.13)), 0, 0, 360, 1.0, -1)
    naso_suppress = cv2.GaussianBlur(naso_suppress, (41, 41), 0)

    # Nasolabial bölgesinde shadow ve highlight sıfırlanır
    shadow_map    = shadow_map    * (1 - naso_suppress)
    highlight_map = highlight_map * (1 - naso_suppress)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    # FIX: shadow/highlight aralığı genişletildi → %30 slider'da bile belli olsun
    lab[:,:,0] -= shadow_map * _scale(t, 8.0, 42.0)
    lab[:,:,0] += highlight_map * _scale(t, 2.0, 14.0)

    result = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

    # Nasolabial fold'u warp tabanlı uygula
    result = _apply_nasolabial_fold(result, landmarks, t, face_w, face_h)

    return result

# ---------------------------------------------------------------------------
# Gray hair
# ---------------------------------------------------------------------------
def _gray_hair(image, landmarks, intensity):
    if landmarks is None or len(landmarks) < 468:
        return image

    h, w = image.shape[:2]
    intensity = float(np.clip(intensity, 0.0, 1.0))

    top = _lm(landmarks, 10)
    chin = _lm(landmarks, 152)
    left = _lm(landmarks, 234)
    right = _lm(landmarks, 454)

    face_w = _dist(left, right)
    face_h = _dist(top, chin)
    cx = (left[0] + right[0]) // 2

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    h_ch, s_ch, v_ch = cv2.split(hsv)

    # Saç bölgesi
    hair_region = np.zeros((h, w), dtype=np.uint8)

    # üst + ön saç çizgisi
    cv2.ellipse(
        hair_region,
        (cx, int(top[1] + face_h * 0.10)),
        (int(face_w * 0.95), int(face_h * 0.58)),
        0,
        165,
        375,
        255,
        -1
    )

    # sol yan saç
    cv2.ellipse(
        hair_region,
        (int(cx - face_w * 0.42), int(top[1] + face_h * 0.25)),
        (int(face_w * 0.24), int(face_h * 0.36)),
        10,
        0,
        360,
        255,
        -1
    )

    # sağ yan saç
    cv2.ellipse(
        hair_region,
        (int(cx + face_w * 0.42), int(top[1] + face_h * 0.25)),
        (int(face_w * 0.24), int(face_h * 0.36)),
        -10,
        0,
        360,
        255,
        -1
    )

    # Yüzü dışla
    face_exclude = np.zeros((h, w), dtype=np.uint8)

    cv2.ellipse(
        face_exclude,
        (cx, int(top[1] + face_h * 0.46)),
        (int(face_w * 0.48), int(face_h * 0.62)),
        0,
        0,
        360,
        255,
        -1
    )

    face_exclude = cv2.dilate(
        face_exclude,
        np.ones((9, 9), np.uint8),
        iterations=1
    )

    # Koyu saç pikselleri
    dark_mask = cv2.inRange(v_ch.astype(np.uint8), 0, 195)
    sat_mask = cv2.inRange(s_ch.astype(np.uint8), 8, 255)

    hair_mask = cv2.bitwise_and(dark_mask, sat_mask)
    hair_mask = cv2.bitwise_and(hair_mask, hair_region)
    hair_mask = cv2.bitwise_and(hair_mask, cv2.bitwise_not(face_exclude))

    kernel = np.ones((5, 5), np.uint8)
    hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_CLOSE, kernel)
    hair_mask = cv2.GaussianBlur(hair_mask, (25, 25), 0)

    base_alpha = hair_mask.astype(np.float32) / 255.0

    # Dipten uca ağırlık
    ys, xs = np.where(hair_mask > 0)
    root_weight = np.zeros((h, w), dtype=np.float32)

    if len(xs) > 0:
        root_y = int(top[1] - face_h * 0.12)
        dist_y = np.clip((ys - root_y) / (face_h * 0.85), 0, 1)

        # Dip 1.0, uç 0.60: uçlar da gri kalır
        values = 1.0 - dist_y * 0.40

        root_weight[ys, xs] = values
        root_weight = cv2.GaussianBlur(root_weight, (31, 31), 0)
        root_weight = np.clip(root_weight, 0.60, 1.0)
    else:
        root_weight = np.ones((h, w), dtype=np.float32)

    hair_t = intensity ** 0.50

    alpha = base_alpha * root_weight
    alpha = np.clip(alpha * hair_t * 1.45, 0, 0.78)

    # Dipler daha beyaz, uçlar gri
    s_ch = s_ch * (1 - alpha * 0.98)
    v_ch = v_ch + alpha * 82

    hsv_new = cv2.merge([
        h_ch,
        np.clip(s_ch, 0, 255),
        np.clip(v_ch, 0, 255)
    ]).astype(np.uint8)

    gray_result = cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR).astype(np.float32)

    # Doğal karışım
    blend = np.clip(alpha * 0.78, 0, 0.60)[:, :, None]

    final = image.astype(np.float32) * (1 - blend) + gray_result * blend

    return np.clip(final, 0, 255).astype(np.uint8)
def _gray_eyebrows(image, landmarks, intensity):
    if landmarks is None or len(landmarks) < 468:
        return image

    h, w = image.shape[:2]
    output = image.copy().astype(np.float32)

    left_brow = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
    right_brow = [336, 296, 334, 293, 300, 285, 295, 282, 283, 276]

    mask = np.zeros((h, w), dtype=np.uint8)

    for brow in [left_brow, right_brow]:
        pts = np.array([_lm(landmarks, i) for i in brow], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 3))
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (9, 9), 0)

    alpha = mask.astype(np.float32) / 255.0
    alpha = np.clip(alpha * intensity * 0.05, 0, 0.05)
    alpha = alpha[:, :, None]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32)

    silver = np.full_like(image, (120, 125, 130), dtype=np.float32)
    target = gray_bgr * 0.80 + silver * 0.20

    result = output * (1 - alpha) + target * alpha

    return np.clip(result, 0, 255).astype(np.uint8)

# ---------------------------------------------------------------------------
# Sagging — jowl + yanak deflasyon + gıdı
# ---------------------------------------------------------------------------
def _apply_sagging(image, landmarks, intensity_raw, face_mask):
    if landmarks is None or len(landmarks) < 468:
        return image

    t = _remap_intensity(intensity_raw)
    h, w = image.shape[:2]

    def p(idx):
        x,y=landmarks[idx]; return int(x),int(y)

    left_face  = p(234); right_face = p(454)
    chin       = p(152); nose_pt    = p(1)
    face_w     = abs(right_face[0]-left_face[0])
    face_h     = abs(chin[1]-nose_pt[1])
    cx         = (left_face[0]+right_face[0])//2
    yy, xx     = np.indices((h,w),dtype=np.float32)

    face_soft = cv2.GaussianBlur(face_mask.astype(np.float32)/255.0, (31,31), 0)

    # ── 1. Jowl sarkması ─────────────────────────────────────────────────────
    jowl_mask = np.zeros((h,w),dtype=np.float32)
    jowl_cy   = int(nose_pt[1]+face_h*0.55)
    cv2.ellipse(jowl_mask,(cx,jowl_cy),
                (int(face_w*0.50),int(face_h*0.36)),0,0,360,1.0,-1)
    jowl_mask[:int(nose_pt[1]+face_h*0.30),:]=0
    jowl_mask = cv2.GaussianBlur(jowl_mask,(61,61),0) * face_soft
    map_y_j   = yy - (jowl_mask * _scale(t, 0.0, 2.8))
    result    = cv2.remap(image, xx, map_y_j,
                          interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)

    # ── 2. Yanak deflasyon (elma kemiği hacim kaybı) ─────────────────────────
    # Genç yüzde elma kemiği dışa-öne çıkar; yaşlıda içe-aşağı çöker.
    # Bunu modellemek için: elma kemiği bölgesini hafifçe içe (merkeze) çekiyoruz
    # + aşağı itiyoruz. Bu "çökmüş" yanak görünümü verir.
    deflate_x = np.zeros((h,w),dtype=np.float32)  # yatay bileşen (içe çekme)
    deflate_y = np.zeros((h,w),dtype=np.float32)  # dikey bileşen (aşağı sarkma)

    for side, cx_off in [(-1, int(cx - face_w*0.26)), (1, int(cx + face_w*0.26))]:
        cheek_center_y = int(nose_pt[1] + face_h*0.28)
        em = np.zeros((h,w),dtype=np.float32)
        cv2.ellipse(em, (cx_off, cheek_center_y),
                    (int(face_w*0.24), int(face_h*0.22)), 0, 0, 360, 1.0, -1)
        em = cv2.GaussianBlur(em, (51,51), 0) * face_soft

        # İçe çekme: sol yanak → sağa (+x), sağ yanak → sola (-x)
        deflate_x += em * (-side) * _scale(t, 0.0, 1.8)
        # Aşağı sarkma
        deflate_y += em * _scale(t, 0.0, 2.2)

    # Lateral (dışa) sarkma — alt yanak kenarı dışa-aşağı kayar
    lateral_x = np.zeros((h,w),dtype=np.float32)
    lateral_y = np.zeros((h,w),dtype=np.float32)
    for side, cx_off in [(-1, int(cx - face_w*0.40)), (1, int(cx + face_w*0.40))]:
        lat_cy = int(nose_pt[1] + face_h*0.50)
        em2    = np.zeros((h,w),dtype=np.float32)
        cv2.ellipse(em2, (cx_off, lat_cy),
                    (int(face_w*0.16), int(face_h*0.25)), 0, 0, 360, 1.0, -1)
        em2 = cv2.GaussianBlur(em2, (41,41), 0) * face_soft
        lateral_x += em2 * side  * _scale(t, 0.0, 1.5)   # dışa kayma
        lateral_y += em2          * _scale(t, 0.0, 1.8)   # aşağı kayma

    map_x_c = xx - deflate_x - lateral_x
    map_y_c = yy - deflate_y - lateral_y
    result  = cv2.remap(result, map_x_c, map_y_c,
                        interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT)

    # ── 3. Yanak hacim kaybı renk tonu (hafif mat/solgun) ────────────────────
    cheek_deflate_mask = np.zeros((h,w),dtype=np.float32)
    for side, cx_off in [(-1, int(cx - face_w*0.26)), (1, int(cx + face_w*0.26))]:
        cv2.ellipse(cheek_deflate_mask,
                    (cx_off, int(nose_pt[1]+face_h*0.28)),
                    (int(face_w*0.22), int(face_h*0.20)), 0, 0, 360, 1.0, -1)
    cheek_deflate_mask = cv2.GaussianBlur(cheek_deflate_mask,(51,51),0) * face_soft

    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)
    # Elma kemiği bölgesini hafifçe koyulaştır (hacim kaybı gölgesi)
    lab[:,:,0] -= cheek_deflate_mask * _scale(t, 0.0, 8.0)
    # Renk doygunluğunu düşür (solgun yanak)
    lab[:,:,1] -= cheek_deflate_mask * _scale(t, 0.0, 4.0)

    # ── 4. Gıdı hacim shadingı ───────────────────────────────────────────────
    dc_bright = np.zeros((h,w),dtype=np.float32)
    cv2.ellipse(dc_bright,(cx, int(chin[1]+face_h*0.08)),
                (int(face_w*0.24),int(face_h*0.10)),0,0,360,1.0,-1)
    dc_bright = cv2.GaussianBlur(dc_bright,(41,41),0) * face_soft
    lab[:,:,0] += dc_bright * _scale(t, 1.0, 14.0)

    dc_shadow = np.zeros((h,w),dtype=np.float32)
    cv2.ellipse(dc_shadow,(cx, int(chin[1]+face_h*0.18)),
                (int(face_w*0.26),int(face_h*0.07)),0,0,360,1.0,-1)
    dc_shadow = cv2.GaussianBlur(dc_shadow,(35,35),0) * face_soft
    lab[:,:,0] -= dc_shadow * _scale(t, 2.0, 20.0)

    result = cv2.cvtColor(np.clip(lab,0,255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    return result

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def apply_aging_effect(image, intensity=0.5, landmarks=None):
    if landmarks is None:
        return image

    img = image.copy()

    f_mask = _build_face_mask(image.shape[0], image.shape[1], landmarks)

    img = _gray_hair(img, landmarks, intensity)
    img = _gray_eyebrows(img, landmarks, intensity)
    img = _apply_wrinkles(img, f_mask, intensity, landmarks)
    img = _apply_sagging(img, landmarks, intensity, f_mask)

    return img