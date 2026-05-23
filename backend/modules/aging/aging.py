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
# Intensity calibration
# ---------------------------------------------------------------------------
def _remap_intensity(raw):
    t = float(np.clip(raw, 0.0, 1.0))
    return t ** 0.50

def _scale(intensity, lo, hi):
    return lo + intensity * (hi - lo)

# ---------------------------------------------------------------------------
# Wrinkle height-field
# ---------------------------------------------------------------------------
def _make_wrinkle_heightfield(h, w, landmarks, face_scale, seed=42):
    rng = np.random.default_rng(seed)
    hmap = np.zeros((h, w), dtype=np.float32)
    if landmarks is None: return hmap

    S = face_scale / 250.0
    top = _lm(landmarks, 10)
    bottom = _lm(landmarks, 152)
    left = _lm(landmarks, 234)
    right = _lm(landmarks, 454)
    face_w = _dist(left, right)
    face_h = _dist(top, bottom)
    cx = (left[0] + right[0]) // 2

    n_c = _perlin(h, w, scale=1.8, seed=seed)
    n_f = _perlin(h, w, scale=0.5, seed=seed+7)
    n_m = _perlin(h, w, scale=0.2, seed=seed+13)

    def _ridge(p1, p2, half_w, height, jitter=0.0, fade_left=0.15, fade_right=0.85):
        L = max(1, int(_dist(p1, p2)))
        t = np.linspace(0, 1, L + 1)
        env = np.where(t < fade_left, t / (fade_left + 1e-6),
              np.where(t > fade_right, (1-t) / (1 - fade_right + 1e-6), 1.0)) ** 1.8
        fade_arr = env * height
        dx = (p2[0]-p1[0])/L; dy = (p2[1]-p1[1])/L
        nx, ny = -dy, dx
        f1 = rng.uniform(1.5, 3.5); f2 = rng.uniform(4.5, 8.0)
        wave = (np.sin(t*np.pi*f1)*0.6 + np.sin(t*np.pi*f2)*0.4) * jitter
        sigma = max(0.5, half_w / 1.6)
        r_range = np.arange(-int(half_w*3), int(half_w*3)+1)
        gauss = np.exp(-(r_range**2) / (2*sigma**2))
        for bx, by, f, wv in zip(p1[0] + (p2[0]-p1[0])*t, p1[1] + (p2[1]-p1[1])*t, fade_arr, wave):
            cx_ = bx + nx*wv; cy_ = by + ny*wv
            px = np.round(cx_ + nx*r_range).astype(int)
            py = np.round(cy_ + ny*r_range).astype(int)
            ok = (px>=0)&(px<w)&(py>=0)&(py<h)
            py_v=py[ok]; px_v=px[ok]
            tex = n_c[py_v,px_v]*0.45 + n_f[py_v,px_v]*0.35 + n_m[py_v,px_v]*0.20
            brk = np.clip(tex*1.8 - 0.20, 0, 1)**0.75
            np.add.at(hmap, (py_v, px_v), gauss[ok]*f*brk)

    w_px = max(0.8, 1.2*S)

    # ── 1. ALIN — DÜZELTİLDİ ───────────────────────────────────────────────
    n_f_lines = rng.integers(4, 7)
    forehead_top = top[1] + face_h * 0.04
    forehead_bot = top[1] + face_h * 0.22
    for i in range(n_f_lines):
        cy = int(forehead_top + (i+0.5)*(forehead_bot-forehead_top)/n_f_lines)
        for _ in range(rng.integers(2,4)):
            seg_w = face_w * rng.uniform(0.22, 0.38)
            cx_ = left[0] + face_w*0.5 + rng.uniform(-face_w*0.18, face_w*0.18) # cx ezilmesin
            sx = int(cx_ - seg_w/2 + rng.uniform(-4,4))
            ex = int(cx_ + seg_w/2 + rng.uniform(-4,4))
            sy = cy + rng.integers(-int(face_h*0.012), int(face_h*0.012))
            ey = sy + rng.integers(-2, 2)
            h_seg = rng.uniform(0.24, 0.42)
            hw = w_px * rng.uniform(1.4, 1.9)
            _ridge((sx, sy), (ex, ey), hw, h_seg,
                   jitter=face_w * rng.uniform(0.012, 0.02),
                   fade_left=0.15, fade_right=0.85)
    # fazladan uzun çizgi silindi

    # Glabella
    brow_mid_y = int(top[1] + face_h * 0.18) # forehead_band hatası düzeltildi
    for k in range(rng.integers(2, 4)):
        offset = rng.integers(-int(face_w*0.05), int(face_w*0.05))
        bx_ = cx + offset
        length = face_h * rng.uniform(0.06, 0.10)
        angle = np.radians(rng.uniform(-98, -82))
        _ridge((bx_, brow_mid_y),
               (int(bx_ + np.cos(angle)*length),
                int(brow_mid_y + np.sin(angle)*length)),
               w_px * rng.uniform(0.8, 1.1),
               rng.uniform(0.35, 0.55),
               jitter=face_w * 0.003,
               fade_left=0.10, fade_right=rng.uniform(0.68, 0.88))

    # ── 2. Göz çevresi ───────────────────────────────────────────────────────
    for eye_idx, sign in [(33,-1),(263,1)]:
        ex, ey = _lm(landmarks, eye_idx)
        for k in range(6):
            angle = np.radians(-50 + k*22)
            length = face_w * rng.uniform(0.09, 0.15)
            _ridge((ex, ey), (int(ex + sign*np.cos(angle)*length), int(ey + np.sin(angle)*length)),
                   w_px*0.9, rng.uniform(0.75, 1.05), jitter=face_w*0.005)
        _ridge((int(ex-sign*face_w*0.10), int(ey-face_h*0.01)), (int(ex+sign*face_w*0.10), int(ey-face_h*0.01)),
               w_px*0.8, rng.uniform(0.28, 0.42), jitter=face_w*0.003)
        for k in range(3):
            oy = face_h*(0.030+k*0.018)
            _ridge((int(ex-sign*face_w*0.10), int(ey+oy)), (int(ex+sign*face_w*0.10), int(ey+oy+face_h*0.012)),
                   w_px*0.75, rng.uniform(0.32, 0.52), jitter=face_w*0.003)
        bc = (ex, int(ey+face_h*0.06))
        for _ in range(7):
            ang = rng.uniform(0, np.pi); l = face_w*rng.uniform(0.04, 0.09)
            _ridge(bc, (int(bc[0]+np.cos(ang)*l), int(bc[1]+np.sin(ang)*l*0.4)),
                   w_px*0.65, rng.uniform(0.16, 0.28), jitter=face_w*0.003)

    # ── 3. Nasolabial ────────────────────────────────────────────────────────
    nose_l = _lm(landmarks, 102); nose_r = _lm(landmarks, 331)
    mouth_l = _lm(landmarks, 61); mouth_r = _lm(landmarks, 291)
    for (np_pt, mp_pt), fold_sign in [((nose_l, mouth_l), -1), ((nose_r, mouth_r), 1)]:
        _ridge(np_pt, mp_pt, w_px*4.5, rng.uniform(0.08, 0.14), jitter=face_w*0.004)
        n_branches = rng.integers(4, 7)
        for _ in range(n_branches):
            t_r = rng.uniform(0.15, 0.85)
            bx_ = int(np_pt[0] + (mp_pt[0]-np_pt[0])*t_r)
            by_ = int(np_pt[1] + (mp_pt[1]-np_pt[1])*t_r)
            angle = np.radians(rng.uniform(30, 70))
            blen = face_w * rng.uniform(0.04, 0.09)
            end_x = int(bx_ + fold_sign * np.cos(angle) * blen)
            end_y = int(by_ + np.sin(angle) * blen * rng.uniform(0.3, 0.8))
            _ridge((bx_, by_), (end_x, end_y), w_px * rng.uniform(0.6, 1.0), rng.uniform(0.25, 0.45),
                   jitter=face_w * 0.003, fade_left=0.05, fade_right=rng.uniform(0.55, 0.80))

    # ── 4. Marionette ────────────────────────────────────────────────────────
    chin_pt = _lm(landmarks, 152)
    for m_idx in [61, 291]:
        mouth = _lm(landmarks, m_idx)
        p2 = (mouth[0]+rng.integers(-4,4), int(mouth[1]+(chin_pt[1]-mouth[1])*0.88))
        _ridge(mouth, p2, w_px*4.0, rng.uniform(0.08, 0.14), jitter=face_w*0.004)

    # ── 5. Yanak ─────────────────────────────────────────────────────────────
    for side, sign in [(-1, -1), (1, 1)]:
        cheek_cx = int(cx + sign * face_w * 0.30)
        cheek_cy = int(top[1] + face_h * 0.40)
        n_ch = rng.integers(3, 6)
        for k in range(n_ch):
            angle = np.radians(rng.uniform(8, 50))
            length = face_w * rng.uniform(0.06, 0.12)
            start = (cheek_cx + sign * rng.integers(-int(face_w*0.03), int(face_w*0.03)),
                      cheek_cy + rng.integers(-int(face_h*0.04), int(face_h*0.04)))
            end = (int(start[0] + sign * np.cos(angle) * length), int(start[1] + np.sin(angle) * length))
            _ridge(start, end, w_px * rng.uniform(0.6, 1.0), rng.uniform(0.28, 0.48),
                   jitter=face_w * 0.004, fade_left=rng.uniform(0.05, 0.15), fade_right=rng.uniform(0.60, 0.88))
    for side, sign in [(-1, -1), (1, 1)]:
        cheek_light_cx = int(cx + sign * face_w * 0.22)
        cheek_light_cy = int(top[1] + face_h * 0.38)
        for k in range(4):
            ang = rng.uniform(-15, 25); length = face_w * rng.uniform(0.07, 0.11)
            sx = cheek_light_cx + rng.integers(-5,5); sy = cheek_light_cy + int(k*face_h*0.04)
            ex = int(sx + sign * np.cos(np.radians(ang)) * length)
            ey = int(sy + np.sin(np.radians(ang)) * length * 0.3)
            _ridge((sx, sy), (ex, ey), w_px * 0.7, rng.uniform(0.18, 0.28), jitter=face_w * 0.003, fade_left=0.15, fade_right=0.75)
        jowl_cx = int(cx + sign * face_w * 0.32); jowl_cy = int(top[1] + face_h * 0.62)
        n_jowl = rng.integers(2, 5)
        for k in range(n_jowl):
            angle = np.radians(rng.uniform(5, 35)); length = face_w * rng.uniform(0.08, 0.16)
            start = (jowl_cx + sign * rng.integers(-int(face_w*0.04), int(face_w*0.04)), jowl_cy + rng.integers(-int(face_h*0.03), int(face_h*0.03)))
            end = (int(start[0] + sign * np.cos(angle) * length), int(start[1] + np.sin(angle) * length))
            _ridge(start, end, w_px * rng.uniform(0.8, 1.2), rng.uniform(0.30, 0.50), jitter=face_w * 0.005,
                   fade_left=rng.uniform(0.08, 0.18), fade_right=rng.uniform(0.65, 0.90))

    # ── 6. Boyun ─────────────────────────────────────────────────────────────
    neck_top_y = int(chin_pt[1] + face_h * 0.08); neck_w_half = int(face_w * 0.30)
    n_neck = rng.integers(4, 6)
    for i in range(n_neck):
        ny_off = int(face_h * 0.075 * i)
        lx_ = cx - neck_w_half + rng.integers(-10, 10); rx_ = cx + neck_w_half + rng.integers(-10, 10)
        ly_ = neck_top_y + ny_off + rng.integers(-5, 5); ry_ = neck_top_y + ny_off + rng.integers(-5, 5)
        p1 = (lx_, ly_); p2 = (rx_, ry_)
        total_w_n = abs(rx_ - lx_); n_segs_n = rng.integers(2, 4)
        seg_starts_n = sorted(rng.uniform(0.0, 0.80, n_segs_n)); seg_lens_n = rng.uniform(0.20, 0.50, n_segs_n)
        for s_s, s_l in zip(seg_starts_n, seg_lens_n):
            s_e = min(s_s + s_l, 1.0)
            if s_e - s_s < 0.10: continue
            sxn = int(lx_ + s_s * total_w_n); exn = int(lx_ + s_e * total_w_n)
            syn = ly_ + rng.integers(-4, 4); eyn = ry_ + rng.integers(-4, 4)
            _ridge((sxn, syn), (exn, eyn),
            w_px * rng.uniform(2.8, 3.4), # daha geniş = yumuşak
            rng.uniform(0.21, 0.31), 
                   jitter=face_w * 0.012, fade_left=rng.uniform(0.08, 0.18), fade_right=rng.uniform(0.72, 0.94))
        n_rays = rng.integers(3, 5)
        for _ in range(n_rays):
            t_r = rng.uniform(0.12, 0.88); rx = int(p1[0] + (p2[0]-p1[0])*t_r); ry = int(p1[1] + (p2[1]-p1[1])*t_r)
            rlen = face_h * rng.uniform(0.018, 0.045); ang = np.radians(rng.uniform(72, 108))
            _ridge((rx, ry), (int(rx + np.cos(ang)*rlen), int(ry + np.sin(ang)*rlen)),
                   w_px * rng.uniform(0.7, 1.0), rng.uniform(0.28, 0.48), jitter=face_w * 0.004,
                   fade_left=0.10, fade_right=rng.uniform(0.62, 0.88))

    pk = hmap.max()
    if pk > 1e-6: hmap = hmap / pk * 1.05
    return hmap

# ---------------------------------------------------------------------------
# Nasolabial fold gölgesi — ince gradyan, ağız korumalı
# ---------------------------------------------------------------------------
def _apply_nasolabial_fold(image, landmarks, t, face_w, face_h):
    h, w = image.shape[:2]
    nose_l = _lm(landmarks, 102); nose_r = _lm(landmarks, 331)
    mouth_l = _lm(landmarks, 61); mouth_r = _lm(landmarks, 291)
    shadow_fold = np.zeros((h, w), dtype=np.float32)
    for (nx_, ny_), (mx_, my_) in [(nose_l, mouth_l), (nose_r, mouth_r)]:
        thickness = max(1, int(face_w * 0.035))
        cv2.line(shadow_fold, (nx_, ny_), (mx_, my_), 1.0, thickness)
    shadow_fold = cv2.GaussianBlur(shadow_fold, (0, 0), sigmaX=face_w * 0.050)
    shadow_fold = np.clip(shadow_fold, 0, 1)
    mouth_protect = np.zeros((h, w), dtype=np.float32)
    for mx_, my_ in [mouth_l, mouth_r]:
        cv2.circle(mouth_protect, (mx_, my_), int(face_w * 0.12), 1.0, -1)
    mouth_protect = cv2.GaussianBlur(mouth_protect, (0, 0), sigmaX=face_w * 0.07)
    shadow_fold = shadow_fold * (1.0 - np.clip(mouth_protect, 0, 1))
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:,:,0] -= shadow_fold * _scale(t, 0.0, 4.0)
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
# ---------------------------------------------------------------------------
# Wrinkle application
# ---------------------------------------------------------------------------
def _apply_wrinkles(image, face_mask, intensity_raw, landmarks):
    t = _remap_intensity(intensity_raw)
    t = np.clip(t, 0, 0.92)
    h, w = image.shape[:2]
    face_scale = _dist(_lm(landmarks, 234), _lm(landmarks, 454))
    top = _lm(landmarks, 10); bottom = _lm(landmarks, 152)
    left = _lm(landmarks, 234); right = _lm(landmarks, 454)
    face_w = _dist(left, right); face_h = _dist(top, bottom)
    cx_face = (left[0] + right[0]) // 2; chin_pt = _lm(landmarks, 152)
    face_soft = cv2.GaussianBlur(face_mask.astype(np.float32)/255.0, (31,31), 0)
    neck_mask = np.zeros((h, w), dtype=np.float32)
    neck_y = int(chin_pt[1] + face_h * 0.10)
    cv2.ellipse(neck_mask, (cx_face, neck_y), (int(face_w * 0.30), int(face_h * 0.20)), 0, 0, 360, 1.0, -1)
    neck_mask[:int(chin_pt[1] + face_h * 0.02), :] = 0
    neck_mask[int(chin_pt[1] + face_h * 0.45):, :] = 0
    neck_mask = cv2.GaussianBlur(neck_mask, (41, 41), 0)
    H_full = _make_wrinkle_heightfield(h, w, landmarks, face_scale)
    H = H_full * face_soft + H_full * neck_mask * 0.70
    gz_val = _scale(t, 0.55, 0.15)
    gx = cv2.Sobel(H, cv2.CV_32F, 1, 0, ksize=5); gy = cv2.Sobel(H, cv2.CV_32F, 0, 1, ksize=5)
    gz = np.ones_like(H) * gz_val
    nlen = np.sqrt(gx**2 + gy**2 + gz**2) + 1e-6
    nx_ = gx/nlen; ny_ = gy/nlen; nz_ = gz/nlen
    lx, ly, lz = -0.4, -0.6, 0.7
    shading = nx_*lx + ny_*ly + nz_*lz
    presence_thresh = _scale(t, 0.12, 0.008)
    presence = np.clip((H - presence_thresh) * 6.0, 0, 1)
    shadow_map = cv2.GaussianBlur(np.clip(-shading, 0, 1) * presence, (9, 9), 2.0)
    highlight_map = cv2.GaussianBlur(np.clip(shading, 0, 1) * presence, (9, 9), 2.0)

    # --- EKLENDİ: gölgeyi uygula ---
    w_px = max(1.0, face_w / 180.0)
    shadow_soft = cv2.GaussianBlur(shadow_map, (0,0), sigmaX=w_px*0.9)
    highlight_soft = cv2.GaussianBlur(highlight_map, (0,0), sigmaX=w_px*0.7)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:,:,0] -= shadow_soft * _scale(t, 6.0, 78.0)
    lab[:,:,0] += highlight_soft * _scale(t, 3, 28.0)

    result = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    result = _apply_nasolabial_fold(result, landmarks, t, face_w, face_h)
    return result

# ---------------------------------------------------------------------------
# Sagging — warp korundu, ton minimize edildi
# YENİ: yanak lateral sarkma warp + boyun yatay sarkma dalgası
# ---------------------------------------------------------------------------
def _apply_sagging(image, landmarks, intensity_raw):
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

    f_mask    = _build_face_mask(h, w, landmarks)
    face_soft = cv2.GaussianBlur(f_mask.astype(np.float32)/255.0, (31,31), 0)

    # ── 1. Jowl sarkması ─────────────────────────────────────────────────────
    jowl_mask = np.zeros((h,w),dtype=np.float32)
    jowl_cy   = int(nose_pt[1]+face_h*0.55)
    cv2.ellipse(jowl_mask,(cx,jowl_cy),
                (int(face_w*0.50),int(face_h*0.36)),0,0,360,1.0,-1)
    jowl_mask[:int(nose_pt[1]+face_h*0.30),:]=0
    jowl_mask = cv2.GaussianBlur(jowl_mask,(61,61),0) * face_soft

    mouth_protect = np.zeros((h,w),dtype=np.float32)
    for m_idx in [61, 291, 13, 14, 0, 17]:
        mx_, my_ = p(m_idx)
        cv2.circle(mouth_protect, (mx_, my_), int(face_w * 0.13), 1.0, -1)
    mouth_protect = cv2.GaussianBlur(mouth_protect, (0,0), sigmaX=face_w*0.08)
    jowl_mask = jowl_mask * (1.0 - np.clip(mouth_protect, 0, 1))

    map_y_j = yy - (jowl_mask * _scale(t, 0.0, 7.5))
    result  = cv2.remap(image, xx, map_y_j,
                        interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT)

    # ── 2. Yanak deflasyon + YENİ lateral sarkma ─────────────────────────────
    deflate_x = np.zeros((h,w),dtype=np.float32)
    deflate_y = np.zeros((h,w),dtype=np.float32)

    # Üst yanak içe-aşağı çökme (mevcut)
    for side, cx_off in [(-1, int(cx - face_w*0.25)), (1, int(cx + face_w*0.25))]:
        cheek_cy = int(nose_pt[1] + face_h*0.26)
        em = np.zeros((h,w),dtype=np.float32)
        cv2.ellipse(em, (cx_off, cheek_cy),
                    (int(face_w*0.26), int(face_h*0.24)), 0, 0, 360, 1.0, -1)
        em = cv2.GaussianBlur(em, (51,51), 0) * face_soft
        deflate_x += em * (-side) * _scale(t, 0.0, 3.5)
        deflate_y += em * _scale(t, 0.0, 4.2)

    # Alt yanak lateral sarkma — dışa ve aşağı (mevcut)
    for side, cx_off in [(-1, int(cx - face_w*0.38)), (1, int(cx + face_w*0.38))]:
        lat_cy = int(nose_pt[1] + face_h*0.52)
        em2    = np.zeros((h,w),dtype=np.float32)
        cv2.ellipse(em2, (cx_off, lat_cy),
                    (int(face_w*0.18), int(face_h*0.26)), 0, 0, 360, 1.0, -1)
        em2 = cv2.GaussianBlur(em2, (41,41), 0) * face_soft
        deflate_x += em2 * side  * _scale(t, 0.0, 3)
        deflate_y += em2          * _scale(t, 0.0,3.5)

    # YENİ: yanak orta-alt bölge ek aşağı sarkma — kırışıklıkları daha görünür yapar
    for side, cx_off in [(-1, int(cx - face_w*0.28)), (1, int(cx + face_w*0.28))]:
        sag_cy = int(nose_pt[1] + face_h*0.44)
        em3    = np.zeros((h,w),dtype=np.float32)
        cv2.ellipse(em3, (cx_off, sag_cy),
                    (int(face_w*0.20), int(face_h*0.22)), 0, 0, 360, 1.0, -1)
        em3 = cv2.GaussianBlur(em3, (45,45), 0) * face_soft
        deflate_y += em3 * _scale(t, 0.0, 3.5)   # sadece aşağı, yatay değil

    map_x_c = xx - deflate_x
    map_y_c = yy - deflate_y
    result  = cv2.remap(result, map_x_c, map_y_c,
                        interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT)

    # ── 3. Boyun sarkması + YENİ yatay kırışıklık dalgası ────────────────────
    chin_pt  = chin
    neck_sag = np.zeros((h,w),dtype=np.float32)
    neck_cy  = int(chin_pt[1] + face_h * 0.10)
    cv2.ellipse(neck_sag, (cx, neck_cy),
                (int(face_w*0.46), int(face_h*0.28)), 0, 0, 360, 1.0, -1)
    neck_sag[:int(chin_pt[1] - face_h*0.02), :] = 0
    neck_sag = cv2.GaussianBlur(neck_sag, (51, 51), 0)

    # Boyun kırışıklık dalgası: yatay sarkma periyodik olarak dalgalanır
    # → düzgün uniform sarkma yerine doğal kırışıklık şekli oluşur
    neck_wave_y = np.zeros((h,w),dtype=np.float32)
    neck_band_y = int(chin_pt[1] + face_h * 0.06)
    for row in range(neck_band_y, min(h, neck_band_y + int(face_h * 0.38))):
        row_weight = neck_sag[row, :]
        if row_weight.max() < 0.01:
            continue
        # Her satır için sinüzoidal x-offset → boyun kırışıklık dalgası
        freq   = 3.5 + (row - neck_band_y) / (face_h * 0.38) * 2.0
        phase  = (row - neck_band_y) * 0.18
        wave_x = np.sin(np.linspace(0, freq * np.pi, w) + phase)
        neck_wave_y[row, :] = wave_x * row_weight * _scale(t, 0.0, 1.2)

    map_y_n = yy - (neck_sag * _scale(t, 0.0, 2.5)) + neck_wave_y
    result  = cv2.remap(result, xx, map_y_n,
                        interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT)

    # ── 4. Ton ayarları — minimal ─────────────────────────────────────────────
    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)

    cheek_shadow = np.zeros((h,w),dtype=np.float32)
    for side, cx_off in [(-1, int(cx - face_w*0.25)), (1, int(cx + face_w*0.25))]:
        cv2.ellipse(cheek_shadow, (cx_off, int(nose_pt[1]+face_h*0.26)),
                    (int(face_w*0.22), int(face_h*0.20)), 0, 0, 360, 1.0, -1)
    cheek_shadow = cv2.GaussianBlur(cheek_shadow,(51,51),0) * face_soft
    lab[:,:,0] -= cheek_shadow * _scale(t, 0.0, 3.5)

    # Gıdı hafif shading
    dc_bright = np.zeros((h,w),dtype=np.float32)
    cv2.ellipse(dc_bright,(cx, int(chin_pt[1]+face_h*0.10)),
                (int(face_w*0.28),int(face_h*0.12)),0,0,360,1.0,-1)
    dc_bright = cv2.GaussianBlur(dc_bright,(41,41),0)
    lab[:,:,0] += dc_bright * _scale(t, 0.5, 7.0)

    dc_shadow = np.zeros((h,w),dtype=np.float32)
    cv2.ellipse(dc_shadow,(cx, int(chin_pt[1]+face_h*0.24)),
                (int(face_w*0.32),int(face_h*0.10)),0,0,360,1.0,-1)
    dc_shadow = cv2.GaussianBlur(dc_shadow,(45,45),0)
    lab[:,:,0] -= dc_shadow * _scale(t, 1.0, 12.0)

    result = cv2.cvtColor(np.clip(lab,0,255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    return result

def _apply_eye_bags(image, landmarks, intensity_raw):
    if landmarks is None: return image
    t = _remap_intensity(intensity_raw)
    h, w = image.shape[:2]
    top = _lm(landmarks,10); bottom = _lm(landmarks,152)
    left = _lm(landmarks,234); right = _lm(landmarks,454)
    face_w = _dist(left,right); face_h = _dist(top,bottom)
    
    result = image.copy()
    yy, xx = np.indices((h,w), dtype=np.float32)
    noise = _perlin(h, w, scale=0.8, seed=7) # dağınıklık için
    
    for eye_idx in [33, 263]:
        ex, ey = _lm(landmarks, eye_idx)
        bag = np.zeros((h,w), np.float32)
        cv2.ellipse(bag, (ex, int(ey + face_h*0.058)),
                    (int(face_w*0.128), int(face_h*0.038)), 0,0,360,1.0,-1)
        bag = cv2.GaussianBlur(bag, (27,27), 0)
        bag = bag * (0.65 + 0.35*noise) # tek renk olmasın
        
        # şişkinlik azaltıldı
        map_y = yy + bag * _scale(t, 0.0, 3.2)
        result = cv2.remap(result, xx, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[:,:,0] -= bag * _scale(t, 0.0, 10.0)  # 15 → 10
        lab[:,:,1] += bag * _scale(t, 0.0, 2.2)   # hafif kızıllık (mor yerine)
        lab[:,:,2] -= bag * _scale(t, 0.0, 2.5)   # 7 → 2.5 (mavilik az)
        result = cv2.cvtColor(np.clip(lab,0,255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    return result

def _apply_age_spots(image, landmarks, intensity_raw):
    t = _remap_intensity(intensity_raw)
    if t < 0.30: # gençte yok
        return image
    h, w = image.shape[:2]
    top = _lm(landmarks,10); bottom = _lm(landmarks,152)
    left = _lm(landmarks,234); right = _lm(landmarks,454)
    face_w = _dist(left,right); face_h = _dist(top,bottom)

    rng = np.random.default_rng(77)
    spots = np.zeros((h,w), np.float32)
    n_spots = int(_scale(t, 0, 20)) # 14 → 20, daha fazla

    for _ in range(n_spots):
        # yanak, alın, şakak bölgesi
        x = rng.integers(int(left[0]+face_w*0.12), int(right[0]-face_w*0.12))
        y = rng.integers(int(top[1]+face_h*0.18), int(bottom[1]-face_h*0.22))
        r = rng.integers(2, max(4, int(face_w*0.018))) # biraz daha büyük
        cv2.circle(spots, (x,y), r, 1.0, -1)

    # kenarları yumuşat + dağınıklık
    spots = cv2.GaussianBlur(spots, (0,0), sigmaX=face_w*0.011)
    spots = spots * (0.7 + 0.3*_perlin(h,w,scale=1.2,seed=11))

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    # ten renginden biraz koyu, kahverengimsi
    lab[:,:,0] -= spots * _scale(t, 0, 12) # 7 → 12 (koyulaşma)
    lab[:,:,1] += spots * _scale(t, 0, 6) # 3 → 6 (kızıllık)
    lab[:,:,2] += spots * _scale(t, 0, 11) # 4 → 11 (sarılık = kahve)

    return cv2.cvtColor(np.clip(lab,0,255).astype(np.uint8), cv2.COLOR_LAB2BGR)

def _gray_hair(image, landmarks, intensity):
    if landmarks is None: return image
    t = _remap_intensity(intensity)
    h, w = image.shape[:2]
    top = _lm(landmarks,10); chin = _lm(landmarks,152)
    left = _lm(landmarks,234); right = _lm(landmarks,454)
    fw = _dist(left,right); fh = _dist(top,chin)
    cx = (left[0]+right[0])//2

    # 1) SAÇ MASKESİ – GrabCut
    face = _build_face_mask(h,w,landmarks)
    face_bg = cv2.dilate(face, np.ones((17,17),np.uint8), 1)

    mask = np.full((h,w), cv2.GC_PR_BGD, np.uint8)
    mask[face_bg>0] = cv2.GC_BGD

    # saç tohumları – 3 nokta (tepe, sol, sağ)
    pts = [
        (cx, int(top[1] - fh*0.25)),
        (int(left[0]+fw*0.08), int(top[1]+fh*0.05)),
        (int(right[0]-fw*0.08), int(top[1]+fh*0.05))
    ]
    for x,y in pts:
        if 0<=y<h and 0<=x<w:
            mask[max(0,y-6):y+6, max(0,x-6):x+6] = cv2.GC_FGD

    bgd = np.zeros((1,65),np.float64); fgd = np.zeros((1,65),np.float64)
    cv2.grabCut(image, mask, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
    hair = np.where((mask==cv2.GC_FGD)|(mask==cv2.GC_PR_FGD),1,0).astype(np.float32)
    hair = cv2.GaussianBlur(hair,(9,9),0)

    # 2) SAÇIN KENDİ RENGİNDEN
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    H,S,V = cv2.split(hsv)
    g = np.clip(t*1.12,0,1); wht = np.clip((t-0.55)*2.6,0,1)
    f = hair * (0.55*g + 0.62*wht)
    S = S*(1-f*0.74); V = V + (255-V)*f*0.26
    out = cv2.cvtColor(np.dstack([H,np.clip(S,0,255),np.clip(V,0,255)]).astype(np.uint8), cv2.COLOR_HSV2BGR)

    # kaş
    brow = np.zeros((h,w),np.uint8)
    for p in [[70,63,105,66,107],[336,296,334,293,300]]:
        cv2.fillPoly(brow,[np.array([_lm(landmarks,i) for i in p],np.int32)],255)
    brow = cv2.GaussianBlur(brow,(7,7),0).astype(np.float32)/255*np.clip(t*0.7,0,1)
    hb = cv2.cvtColor(out,cv2.COLOR_BGR2HSV).astype(np.float32)
    Hb,Sb,Vb = cv2.split(hb); Sb*=(1-brow*0.42); Vb+=(255-Vb)*brow*0.22
    out = cv2.cvtColor(np.dstack([Hb,np.clip(Sb,0,255),np.clip(Vb,0,255)]).astype(np.uint8),cv2.COLOR_HSV2BGR)
    return out
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _apply_double_chin(image, landmarks, intensity_raw):
    t = _remap_intensity(intensity_raw)
    h, w = image.shape[:2]
    def p(i): x,y = landmarks[i]; return int(x),int(y)
    chin = p(152); left = p(234); right = p(454); nose = p(1)
    face_w = abs(right[0]-left[0]); face_h = abs(chin[1]-nose[1])
    cx = (left[0]+right[0])//2
    yy, xx = np.indices((h,w), dtype=np.float32)
    chin_mask = np.zeros((h,w), dtype=np.float32)
    cy = int(chin[1] + face_h*0.16)
    cv2.ellipse(chin_mask, (cx, cy), (int(face_w*0.28), int(face_h*0.16)), 0,0,360,1.0,-1) 
    chin_mask = cv2.GaussianBlur(chin_mask, (51,51), 0)
    bulge_y = chin_mask * _scale(t, 0.0, 14.5)
    bulge_x = (xx - cx) / (face_w*0.5 + 1e-6)
    map_y = yy + bulge_y
    map_x = xx + bulge_x * chin_mask * _scale(t, 0.0, 6.8)
    warped = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB).astype(np.float32)
    shadow = np.zeros((h,w), dtype=np.float32)
    cv2.ellipse(shadow, (cx, int(chin[1]+face_h*0.07)), (int(face_w*0.28), int(face_h*0.09)),0,0,360,1.0,-1)
    shadow = cv2.GaussianBlur(shadow, (35,35), 0)
    lab[:,:,0] -= shadow * _scale(t, 0.0, 32.0)
    return cv2.cvtColor(np.clip(lab,0,255).astype(np.uint8), cv2.COLOR_LAB2BGR)

def apply_aging_effect(image, intensity=0.5, landmarks=None):
    if landmarks is None:
        return image
    img = image.copy()
    f_mask = _build_face_mask(image.shape[0], image.shape[1], landmarks)
    img = _apply_wrinkles(img, f_mask, intensity, landmarks)
    img = _apply_eye_bags(img, landmarks, intensity)  
    img = _apply_age_spots(img, landmarks, intensity)
    img = _apply_sagging(img, landmarks, intensity)
    img = _apply_double_chin(img, landmarks, intensity) # YENİ
    img = _gray_hair(img, landmarks, intensity)
    return img