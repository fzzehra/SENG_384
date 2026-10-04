import cv2
import numpy as np


def _hex_to_bgr(color_hex: str) -> tuple[int, int, int]:
    color_hex = (color_hex or "#241815").strip().lstrip("#")
    if len(color_hex) != 6:
        color_hex = "241815"
    r = int(color_hex[0:2], 16)
    g = int(color_hex[2:4], 16)
    b = int(color_hex[4:6], 16)
    return (b, g, r)


def _pt(landmarks, idx: int, w: int, h: int) -> np.ndarray:
    p = landmarks[idx]

    if isinstance(p, dict):
        x, y = p["x"], p["y"]
    elif hasattr(p, "x") and hasattr(p, "y"):
        x, y = p.x, p.y
    else:
        x, y = p[0], p[1]

    if x <= 1.5 and y <= 1.5:
        x, y = x * w, y * h

    return np.array([x, y], dtype=np.float32)


# MediaPipe FaceMesh yüz oval indeksleri (sıralı çevre)
_FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
              397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
              172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

# Çene hattı: görüntüde soldan sağa
_JAW_PATH = [132, 58, 172, 136, 150, 149, 176, 148, 152,
             377, 400, 378, 379, 365, 397, 288, 361]

# Dış dudak konturu
_LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
               409, 270, 269, 267, 0, 37, 39, 40, 185]


def _odd(n: float, minimum: int = 3) -> int:
    n = max(minimum, int(n))
    return n if n % 2 == 1 else n + 1


def _ellipse_kernel(k: int):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _lowfreq_noise(h: int, w: int, cells_x: int, cells_y: int, rng) -> np.ndarray:
    """Düşük frekanslı gürültü: komşu kılların aynı yöne yatmasını (öbek) sağlar."""
    g = rng.normal(0, 1, (cells_y, cells_x)).astype(np.float32)
    return cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)


def apply_beard_effect(image, landmarks, intensity=0.8, color_hex="#241815", **kwargs):
    out = image.copy()
    h, w = out.shape[:2]

    intensity = float(np.clip(intensity, 0.0, 1.0))
    if intensity <= 0.0:
        return out
    base_bgr = np.array(_hex_to_bgr(color_hex), dtype=np.float32)

    def P(i):
        return _pt(landmarks, i, w, h)

    chin = P(152)
    lower_lip = P(17)
    lip_top = P(0)
    nose_base = P(2)
    m_a, m_b = sorted([P(61), P(291)], key=lambda p: float(p[0]))   # ağız köşeleri (sol, sağ)
    n_a, n_b = sorted([P(98), P(327)], key=lambda p: float(p[0]))   # burun delikleri (sol, sağ)

    face_width = float(np.linalg.norm(P(454) - P(234)))
    beard_height = max(20.0, float(chin[1] - lower_lip[1]))

    if face_width < 20 or beard_height < 6:
        return out

    # ───────────────────────── 1) Bölgeler ─────────────────────────────────────
    # Sakal: çene hattı + ağız köşelerine kadar yanaklar
    jaw_pts = [P(i) for i in _JAW_PATH]
    m_a_o = np.array([m_a[0] - face_width * 0.10, m_a[1] - beard_height * 0.10])
    m_b_o = np.array([m_b[0] + face_width * 0.10, m_b[1] - beard_height * 0.10])
    beard_poly = np.array(jaw_pts + [m_b_o, lip_top, m_a_o], dtype=np.float32)

    # Bıyık: burun altı ile üst dudak arası
    m_a_i = np.array([m_a[0] - face_width * 0.03, m_a[1]])
    m_b_i = np.array([m_b[0] + face_width * 0.03, m_b[1]])
    mous_poly = np.array([n_a, nose_base, n_b, m_b_i, lip_top, m_a_i], dtype=np.float32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [beard_poly.astype(np.int32)], 255)
    mous = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mous, [mous_poly.astype(np.int32)], 255)
    mask = np.maximum(mask, mous)

    # Dudak: kıl çıkmasın, kıllar dudağın üstüne binmesin
    lips = np.array([P(i) for i in _LIPS_OUTER], dtype=np.int32)
    lip_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(lip_mask, [lips], 255)
    lip_dil = cv2.dilate(lip_mask, _ellipse_kernel(_odd(face_width * 0.02)))
    mask[lip_dil > 0] = 0
    lk = _odd(face_width * 0.02)
    lip_keep = 1.0 - cv2.GaussianBlur(lip_dil, (lk, lk), 0).astype(np.float32) / 255.0

    # Yüz ovali: gölge içeride kalır, kıllar sadece çok az dışarı taşabilir
    face_poly = np.array([P(i) for i in _FACE_OVAL], dtype=np.int32)
    face_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(face_mask, [face_poly], 255)
    fk = _odd(face_width * 0.04, 5)
    face_in = cv2.erode(face_mask, _ellipse_kernel(_odd(face_width * 0.02)))
    face_in_soft = cv2.GaussianBlur(face_in, (fk, fk), 0).astype(np.float32) / 255.0
    face_out = cv2.dilate(face_mask, _ellipse_kernel(_odd(face_width * 0.03)))
    face_out_soft = cv2.GaussianBlur(face_out, (fk, fk), 0).astype(np.float32) / 255.0

    # Yoğunluk profili: çene/çene altı sık, yanakta yukarı doğru seyrek
    y_top = float(min(m_a[1], m_b[1]) - beard_height * 0.5)
    y_bot = float(chin[1])
    yy = np.arange(h, dtype=np.float32)[:, None]
    t = np.clip((yy - y_top) / max(1.0, y_bot - y_top), 0.0, 1.0)
    prof_col = 0.15 + 0.85 * (t ** 0.9)                    # (h, 1)

    root_k = _odd(face_width * 0.03, 3)
    root01 = cv2.GaussianBlur(mask, (root_k, root_k), 0).astype(np.float32) / 255.0
    root01 *= face_in_soft
    D = root01 * prof_col
    D = np.where(mous > 0, root01 * 0.95, D)                # bıyık: profilden bağımsız, sık

    rng = np.random.default_rng(42)
    out_f = out.astype(np.float32)

    # ───────────────────────── 2) Sakal kütlesi (asıl doğallığı veren katman) ───
    # Gerçek sakalda önce koyu, dokulu bir kütle görülür; tek tek kıllar sadece kenarlarda seçilir.
    # Ten dokusu kısmen korunur, üstüne ince tanecikli (grain) bir doku biner.
    sk = _odd(face_width * 0.08, 5)
    mass = cv2.GaussianBlur(mask, (sk, sk), 0).astype(np.float32) / 255.0
    # Düzensiz büyüme: yer yer seyrek, yer yer sık (kenarın "kesilmiş" görünmesini engeller)
    patch_lo = cv2.resize(
        rng.normal(0, 1, (max(3, int(h / (face_width * 0.18))),
                          max(3, int(w / (face_width * 0.18))))).astype(np.float32),
        (w, h), interpolation=cv2.INTER_CUBIC)
    patch_lo /= float(patch_lo.std()) + 1e-6
    mass = (mass * face_in_soft * lip_keep * np.where(mous > 0, 0.85, prof_col)
            * np.clip(1.0 + 0.22 * patch_lo, 0.6, 1.3))

    fine = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 0.7)
    fine /= float(fine.std()) + 1e-6
    coarse = cv2.resize(
        rng.normal(0, 1, (max(3, h // 5), max(3, w // 5))).astype(np.float32),
        (w, h), interpolation=cv2.INTER_CUBIC)
    coarse /= float(coarse.std()) + 1e-6
    grain = np.clip(1.0 + 0.22 * fine + 0.18 * coarse, 0.5, 1.6)

    a = np.clip(mass * (0.18 + 0.30 * intensity) * grain, 0.0, 0.9)[:, :, None]
    lift = np.clip(base_bgr * 1.15 + np.array([2, 4, 8], dtype=np.float32), 0, 255)
    tinted = out_f * 0.20 + lift * 0.80
    out_f = out_f * (1.0 - a) + tinted * a

    # ───────────────────────── 3) Kıllar ───────────────────────────────────────
    ys, xs = np.nonzero(D > 0.02)
    if xs.size > 0:
        wts = D[ys, xs].astype(np.float64)
        total = float(wts.sum())
        prob = wts / total
        d_max = float(D.max()) + 1e-6

        # Süper-örnekleme: ince kıllar için 2x tuval (büyük görsellerde kapalı)
        S = 2 if max(h, w) <= 1400 else 1
        H, W = h * S, w * S
        canvas_c = np.zeros((H, W, 3), dtype=np.uint8)   # önceden alfa ile çarpılmış renk
        canvas_a = np.zeros((H, W), dtype=np.uint8)      # alfa

        # --- Sakal dibi (stubble): çok sayıda küçük nokta ---
        n_st = int(min(300000, total * (1.0 + 2.0 * intensity)))
        pk = rng.choice(xs.size, size=n_st, p=prob)
        sx = np.clip((xs[pk] + rng.uniform(0, 1, n_st)) * S, 0, W - 1).astype(np.int32)
        sy = np.clip((ys[pk] + rng.uniform(0, 1, n_st)) * S, 0, H - 1).astype(np.int32)
        dop = rng.uniform(60, 170, n_st)
        dcol = np.clip(base_bgr[None, :] * rng.uniform(0.6, 1.3, (n_st, 1)), 0, 255)
        canvas_c[sy, sx] = (dcol * (dop / 255.0)[:, None]).astype(np.uint8)
        canvas_a[sy, sx] = dop.astype(np.uint8)

        # --- Kısa kıllar: çok sayıda, düşük opaklık, sadece doku verir ---
        n_hair = int(min(80000, max(300, total * (0.10 + 0.40 * intensity))))
        pk = rng.choice(xs.size, size=n_hair, p=prob)
        ix, iy = xs[pk], ys[pk]
        px = ix + rng.uniform(-0.5, 0.5, n_hair)
        py = iy + rng.uniform(-0.5, 0.5, n_hair)
        is_m = mous[iy, ix] > 0

        # Yön: aşağı; yanakta çeneye doğru hafif içe, bıyıkta ortadan dışa
        cx = float(chin[0])
        side = np.clip((cx - px) / (face_width * 0.5), -1.0, 1.0)
        ang = np.where(is_m, -side * 0.7, side * 0.45)
        cells_x = max(3, int(w / (face_width * 0.07)))
        cells_y = max(3, int(h / (face_width * 0.07)))
        nz = _lowfreq_noise(h, w, cells_x, cells_y, rng)
        ang = ang + 0.40 * nz[iy, ix] + rng.normal(0, 0.15, n_hair)

        # Uzunluk: kısa (kalem çizgisi gibi durmasın); yoğunlukla hafif artar
        l_max = face_width * (0.015 + 0.026 * intensity)
        local_prof = prof_col[iy, 0]
        lengths = l_max * rng.uniform(0.35, 1.0, n_hair) * (0.7 + 0.3 * local_prof)
        lengths = np.where(is_m, lengths * 0.8, lengths)
        guard = rng.random(n_hair) < 0.15                # az sayıda daha uzun, koyu "ana" kıl
        lengths = np.where(guard, lengths * 1.7, lengths)
        lengths = np.maximum(lengths, 1.5)

        bend = rng.normal(0, 0.10, n_hair) * lengths
        dx, dy = np.sin(ang), np.cos(ang)
        x0 = (px * S).astype(np.int32)
        y0 = (py * S).astype(np.int32)
        xm = ((px + dx * lengths * 0.5 + bend) * S).astype(np.int32)
        ym = ((py + dy * lengths * 0.5) * S).astype(np.int32)
        x1 = ((px + dx * lengths + bend * 1.6) * S).astype(np.int32)
        y1 = ((py + dy * lengths) * S).astype(np.int32)

        # Renk: koyu kütlenin üstünde biraz açık tonlu kıllar (kontrast düşük)
        v = rng.uniform(0.7, 2.0, n_hair)
        v = np.where(guard, rng.uniform(0.35, 0.9, n_hair), v)[:, None]   # ana kıllar koyu
        col_root = np.clip(base_bgr[None, :] * v * 0.8 + np.array([2, 4, 8]), 0, 255)
        col_tip = np.clip(base_bgr[None, :] * v * 1.4 + np.array([6, 10, 18]), 0, 255)
        edge = (0.55 + 0.45 * (D[iy, ix] / d_max))
        op = rng.uniform(70, 170, n_hair) * edge
        op = np.where(guard, rng.uniform(120, 200, n_hair) * edge, op)
        op_t = op * 0.6
        pre_root = (col_root * (op / 255.0)[:, None]).astype(np.uint8).tolist()
        pre_tip = (col_tip * (op_t / 255.0)[:, None]).astype(np.uint8).tolist()
        a_root = op.astype(np.uint8).tolist()
        a_tip = op_t.astype(np.uint8).tolist()

        thick = max(1, int(round(face_width * S / 300.0)))
        x0l, y0l, xml, yml, x1l, y1l = (a_.tolist() for a_ in (x0, y0, xm, ym, x1, y1))
        for i in range(n_hair):
            p0 = (x0l[i], y0l[i])
            pm = (xml[i], yml[i])
            p1 = (x1l[i], y1l[i])
            cv2.line(canvas_c, p0, pm, tuple(pre_root[i]), thick, cv2.LINE_AA)
            cv2.line(canvas_a, p0, pm, a_root[i], thick, cv2.LINE_AA)
            cv2.line(canvas_c, pm, p1, tuple(pre_tip[i]), thick, cv2.LINE_AA)
            cv2.line(canvas_a, pm, p1, a_tip[i], thick, cv2.LINE_AA)

        if S > 1:
            canvas_c = cv2.resize(canvas_c, (w, h), interpolation=cv2.INTER_AREA)
            canvas_a = cv2.resize(canvas_a, (w, h), interpolation=cv2.INTER_AREA)
        # Hafif yumuşatma: çizgi keskinliğini al
        canvas_c = cv2.GaussianBlur(canvas_c, (3, 3), 0.6)
        canvas_a = cv2.GaussianBlur(canvas_a, (3, 3), 0.6)

        gain = 0.85 + 0.15 * intensity
        g = (face_out_soft * lip_keep * gain)[:, :, None]
        a_f = canvas_a.astype(np.float32)[:, :, None] / 255.0
        out_f = out_f * (1.0 - a_f * g) + canvas_c.astype(np.float32) * g

    return np.clip(out_f, 0, 255).astype(np.uint8)