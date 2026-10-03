"""Sakal taşma kontrolü - mevcut dosyalarına dokunmaz.

1) Aşağıdaki iki satırı kendi proje yapına göre düzenle:
       BEARD_MODULE : apply_beard_effect fonksiyonunun bulunduğu .py dosyasının adı (uzantısız)
       BEARD_FUNC   : fonksiyonun adı
2) Çalıştır:
       python check_beard.py foto.jpg

Çıktılar:
    check_overlay.png   -> landmark'lar (yüz oval mavi, çene hattı yeşil) görselin üstünde
    check_result.png    -> sakal uygulanmış görsel (intensity 0.8)
    check_changed.png   -> sakalın değiştirdiği pikseller (beyaz) + yüz ovali (mavi çizgi)
Konsolda: sakalın yüz ovali DIŞINDA değiştirdiği piksel sayısı yazar.
"""
import importlib
import sys

import cv2
import numpy as np

from backend.modules.landmark.landmark import detect_landmarks   # landmark.py'nin yeri

BEARD_MODULE = "backend.modules.beard.beard"   # sakal kodunun olduğu beard.py
BEARD_FUNC = "apply_beard_effect"

FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
             397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
             172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
JAW_PATH = [132, 58, 172, 136, 150, 149, 176, 148, 152,
            377, 400, 378, 379, 365, 397, 288, 361]


def main():
    image = cv2.imread(sys.argv[1])
    if image is None:
        print("Görsel okunamadı")
        return
    h, w = image.shape[:2]
    print("görsel boyutu (w x h):", w, "x", h)

    lm = detect_landmarks(image)
    if not lm:
        print("Yüz bulunamadı")
        return
    print("landmark sayısı:", len(lm), "| yüz genişliği (234-454):", abs(lm[454][0] - lm[234][0]), "px")

    # Overlay
    vis = image.copy()
    for i in FACE_OVAL:
        cv2.circle(vis, lm[i], 3, (255, 0, 0), -1)
    for i in JAW_PATH:
        cv2.circle(vis, lm[i], 4, (0, 255, 0), -1)
    cv2.imwrite("check_overlay.png", vis)

    # Sakal
    mod = importlib.import_module(BEARD_MODULE)
    fn = getattr(mod, BEARD_FUNC)
    print("kullanılan modül:", mod.__file__)
    result = fn(image, lm, intensity=0.8, color_hex="#241815")
    cv2.imwrite("check_result.png", result)

    # Yüz ovali dışına taşma ölçümü
    diff = (np.abs(result.astype(np.int16) - image.astype(np.int16)).max(axis=2) > 2)
    oval = np.array([lm[i] for i in FACE_OVAL], dtype=np.int32)
    oval_mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(oval_mask, [oval], 255)
    outside = int(np.count_nonzero(diff & (oval_mask == 0)))
    total = int(np.count_nonzero(diff))
    print(f"değişen piksel: {total} | yüz ovali DIŞINDA değişen: {outside}")

    changed = (diff.astype(np.uint8) * 255)
    changed = cv2.cvtColor(changed, cv2.COLOR_GRAY2BGR)
    cv2.polylines(changed, [oval], True, (255, 0, 0), 1)
    cv2.imwrite("check_changed.png", changed)
    print("Kaydedildi: check_overlay.png, check_result.png, check_changed.png")


if __name__ == "__main__":
    main()