"""
Klasifikasi Kelayakan Penerima Beasiswa BAZNAS Menggunakan Decision Tree
Dataset : Dataset_Beasiswa_BAZNAS_Indonesia.csv (1000 mahasiswa)

PENTING: dataset asli TIDAK punya kolom label. Label "Layak / Tidak Layak"
dibuat dari aturan skor di fungsi hitung_skor() di bawah (kriteria buatan,
bukan keputusan panitia BAZNAS yang sebenarnya). Cantumkan hal ini di makalah.

Cara pakai:
1. pip install pandas scikit-learn matplotlib
2. Taruh file CSV di folder yang sama dengan file ini.
3. python decision_tree_beasiswa_baznas.py
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report)

CSV_PATH = "Dataset_Beasiswa_BAZNAS_Indonesia.csv"
AMBANG_SKOR = 65     # skor minimal agar Layak
IPK_MINIMAL = 3.0    # syarat wajib

# ------------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)
print("Jumlah data :", len(df))
print("Kolom       :", list(df.columns))
print("Data kosong :", df.isna().sum().sum())

# ------------------------------------------------------------------
# 2. MEMBUAT LABEL DARI ATURAN SKOR (maksimal 100 poin)
# ------------------------------------------------------------------
def hitung_skor(r):
    s = 0
    # IPK (maks 30)
    s += 30 if r.IPK >= 3.5 else 20 if r.IPK >= 3.0 else 10
    # Pendapatan keluarga per bulan (maks 25), makin rendah makin tinggi
    if r.Pendapatan_Keluarga < 2_000_000:
        s += 25
    elif r.Pendapatan_Keluarga <= 3_500_000:
        s += 15
    else:
        s += 5
    # Jumlah tanggungan (maks 15)
    s += 15 if r.Jumlah_Tanggungan >= 5 else 10 if r.Jumlah_Tanggungan >= 3 else 5
    # Status orang tua (maks 15)
    s += {"Yatim Piatu": 15, "Yatim": 10, "Piatu": 10, "Lengkap": 0}[r.Status_Orang_Tua]
    # Skor motivasi (maks 10)
    s += 10 if r.Skor_Motivasi >= 80 else 6 if r.Skor_Motivasi >= 65 else 3
    # Aktivitas organisasi (maks 5)
    s += 5 if r.Aktivitas_Organisasi == "Aktif" else 0
    return s

df["Skor"] = df.apply(hitung_skor, axis=1)
df["Status"] = ((df["Skor"] >= AMBANG_SKOR) & (df["IPK"] >= IPK_MINIMAL)).map(
    {True: "Layak", False: "Tidak Layak"})

print("\n=== DISTRIBUSI LABEL ===")
print(df["Status"].value_counts())

# Simpan dataset berlabel (bisa dipakai untuk Excel)
df.to_csv("Dataset_Beasiswa_BAZNAS_berlabel.csv", index=False)

# ------------------------------------------------------------------
# 3. PILIH FITUR
# ------------------------------------------------------------------
# ID, Nama = identitas (tidak dipakai). Skor = dipakai membuat label,
# TIDAK boleh jadi fitur (kalau dipakai, model "mencontek").
fitur_num = ["Pendapatan_Keluarga", "Jumlah_Tanggungan", "IPK", "Semester",
             "Usia", "Skor_Motivasi"]
fitur_kat = ["Status_Orang_Tua", "Aktivitas_Organisasi", "Pekerjaan_Sampingan"]

X = pd.get_dummies(df[fitur_num + fitur_kat], columns=fitur_kat)
y = df["Status"]

# ------------------------------------------------------------------
# 4. SPLIT DATA 80:20
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nData latih: {len(X_train)} | Data uji: {len(X_test)}")

# ------------------------------------------------------------------
# 5. TRAINING DECISION TREE (kriteria entropy = Information Gain)
# ------------------------------------------------------------------
model = DecisionTreeClassifier(criterion="entropy", max_depth=5,
                               random_state=42)
model.fit(X_train, y_train)

print("\n=== ATURAN POHON KEPUTUSAN ===")
print(export_text(model, feature_names=list(X.columns)))

# ------------------------------------------------------------------
# 6. EVALUASI
# ------------------------------------------------------------------
y_pred = model.predict(X_test)
labels = ["Layak", "Tidak Layak"]
cm = confusion_matrix(y_test, y_pred, labels=labels)
print("=== CONFUSION MATRIX ===")
print(pd.DataFrame(cm, index=["Aktual Layak", "Aktual Tidak Layak"],
                   columns=["Prediksi Layak", "Prediksi Tidak Layak"]))
print(f"\nAkurasi data latih: {model.score(X_train, y_train) * 100:.2f}%")
print(f"Akurasi data uji  : {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred, zero_division=0))

# ------------------------------------------------------------------
# 7. FITUR PALING BERPENGARUH
# ------------------------------------------------------------------
penting = pd.Series(model.feature_importances_, index=X.columns)
print("=== TINGKAT KEPENTINGAN FITUR ===")
print(penting.sort_values(ascending=False).round(4).head(10))

# ------------------------------------------------------------------
# 8. VISUALISASI POHON
# ------------------------------------------------------------------
plt.figure(figsize=(26, 12))
plot_tree(model, feature_names=list(X.columns), class_names=model.classes_,
          filled=True, rounded=True, fontsize=7)
plt.title("Decision Tree Kelayakan Penerima Beasiswa BAZNAS")
plt.savefig("pohon_keputusan_baznas.png", dpi=130, bbox_inches="tight")
print("\nGambar pohon disimpan: pohon_keputusan_baznas.png")

# ------------------------------------------------------------------
# 9. PREDIKSI MAHASISWA BARU
# ------------------------------------------------------------------
baru = pd.DataFrame([{
    "Pendapatan_Keluarga": 1_800_000, "Jumlah_Tanggungan": 5, "IPK": 3.6,
    "Semester": 4, "Usia": 20, "Skor_Motivasi": 85,
    "Status_Orang_Tua": "Yatim", "Aktivitas_Organisasi": "Aktif",
    "Pekerjaan_Sampingan": "Tidak Ada"}])
baru_enc = pd.get_dummies(baru, columns=fitur_kat).reindex(
    columns=X.columns, fill_value=0)
print("Prediksi mahasiswa baru:", model.predict(baru_enc)[0])