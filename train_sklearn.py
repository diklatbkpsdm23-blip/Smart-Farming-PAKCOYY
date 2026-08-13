"""
Training classifier Pakcoy (belum_panen vs mendekati_panen) — scikit-learn
============================================================================
Dipakai sebagai model bawaan default di app.py (dimuat otomatis di halaman
"Deteksi Siap Panen (CNN 2D)") karena lingkungan training saat ini tidak
punya TensorFlow. Model ini genuinely dilatih dari fitur warna/tekstur
daun hasil ekstraksi dari foto asli — bukan CNN piksel mentah, tapi hasil
prediksinya nyata (bukan heuristik hardcode seperti mode fallback app.py).

Jika nanti TensorFlow tersedia, jalankan train_cnn2d_pakcoy.py untuk model
CNN 2D asli (lebih akurat), lalu unggah .h5-nya di app.py untuk override.
"""
import os
import json
import pickle
import numpy as np
from PIL import Image

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

from pakcoy_features import extract_features, IMG_SIZE

DATASET_DIR = "dataset"
SEED = 42


def load_dataset(dataset_dir):
    classes = sorted([d for d in os.listdir(dataset_dir)
                       if os.path.isdir(os.path.join(dataset_dir, d))])
    X, y, paths = [], [], []
    for ci, cls in enumerate(classes):
        folder = os.path.join(dataset_dir, cls)
        for fn in sorted(os.listdir(folder)):
            fp = os.path.join(folder, fn)
            try:
                img = Image.open(fp)
                X.append(extract_features(img))
                y.append(ci)
                paths.append(fp)
            except Exception as e:
                print(f"skip {fp}: {e}")
    return np.array(X), np.array(y), classes, paths


def main():
    X, y, classes, paths = load_dataset(DATASET_DIR)
    print(f"Total sampel: {len(X)} | Kelas: {classes}")
    print(f"Distribusi: {[(c, int((y == i).sum())) for i, c in enumerate(classes)]}")

    candidates = {
        "logreg": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0, random_state=SEED)),
        ]),
        "svm_rbf": Pipeline([
            ("scale", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=2.0, gamma="scale", probability=True,
                        class_weight="balanced", random_state=SEED)),
        ]),
        "random_forest": Pipeline([
            ("scale", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=300, max_depth=6,
                                           class_weight="balanced", random_state=SEED)),
        ]),
    }

    # Cross-validation (dataset kecil -> CV lebih jujur daripada 1x holdout)
    n_splits = min(5, int(np.min(np.bincount(y))))
    n_splits = max(2, n_splits)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    best_name, best_score, best_model = None, -1, None
    print(f"\nEvaluasi {n_splits}-fold cross-validation (F1-macro):")
    for name, pipe in candidates.items():
        scores = cross_val_score(pipe, X, y, cv=skf, scoring="f1_macro")
        print(f"  {name:15s}: {scores.mean():.3f} (+/- {scores.std():.3f})  {np.round(scores, 3)}")
        if scores.mean() > best_score:
            best_score, best_name, best_model = scores.mean(), name, pipe

    print(f"\nModel terbaik: {best_name} (F1-macro CV = {best_score:.3f})")

    # Latih ulang model terbaik di seluruh data (untuk dipakai di produksi)
    best_model.fit(X, y)

    # Evaluasi holdout terpisah untuk laporan (stratified split kecil)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)
    report_model = candidates[best_name]
    report_model.fit(X_tr, y_tr)
    y_pred = report_model.predict(X_te)
    print("\nLaporan pada holdout 20%:")
    print(classification_report(y_te, y_pred, target_names=classes, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_te, y_pred))

    # Simpan model final (dilatih di semua data) + metadata
    with open("pakcoy_cnn2d_sklearn.pkl", "wb") as f:
        pickle.dump({"model": best_model, "classes": classes,
                     "feature_fn": "extract_features_v1",
                     "img_size": IMG_SIZE}, f)

    labels_display = [c.replace("_", " ").title() for c in classes]
    with open("class_order.json", "w") as f:
        json.dump(labels_display, f, indent=2, ensure_ascii=False)

    print(f"\nModel disimpan: pakcoy_cnn2d_sklearn.pkl")
    print(f"Urutan kelas (class_order.json): {labels_display}")


if __name__ == "__main__":
    main()
