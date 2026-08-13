"""Ekstraksi fitur warna/tekstur daun Pakcoy dari foto — dipakai bersama
oleh train_sklearn.py (training) dan app.py (prediksi) supaya konsisten."""
import numpy as np

IMG_SIZE = (150, 150)


def extract_features(pil_img):
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).astype(np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

    hsv = np.array(img.convert("HSV")).astype(np.float32)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    green_mask = (g > r) & (g > b * 0.9) & (g > 40)
    coverage = float(green_mask.mean())

    feats = []
    h_hist, _ = np.histogram(h, bins=16, range=(0, 255), density=True)
    feats.extend(h_hist.tolist())
    s_hist, _ = np.histogram(s, bins=8, range=(0, 255), density=True)
    feats.extend(s_hist.tolist())
    feats.extend([
        coverage,
        float(g.mean() / 255.0), float(g.std() / 255.0),
        float(r.mean() / 255.0), float(b.mean() / 255.0),
        float(v.mean() / 255.0), float(v.std() / 255.0),
        float(s.mean() / 255.0),
    ])
    if coverage > 0.02:
        feats.append(float(g[green_mask].mean() / 255.0))
        feats.append(float(((np.maximum(np.maximum(r, g), b) -
                              np.minimum(np.minimum(r, g), b)) / 255.0)[green_mask].mean()))
    else:
        feats.extend([0.0, 0.0])

    gray = np.array(img.convert("L")).astype(np.float32)
    gy, gx = np.gradient(gray)
    feats.append(float(np.sqrt(gx**2 + gy**2).mean() / 255.0))

    return np.array(feats, dtype=np.float32)
