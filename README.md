# Tic-Tac-Toe Endgame - ID3 Karar Ağacı ile Sınıflandırma

Bu proje, Bursa Teknik Üniversitesi Bilgisayar Mühendisliği Bölümü'nde yürütülen BLM0463 Veri Madenciliği dersi kapsamında hazırlanmıştır. 
Proje kapsamında, UCI Machine Learning Repository'den alınan **Tic-Tac-Toe Endgame** veri seti kullanılarak **ID3 (entropy tabanlı)** karar ağacı algoritması ile sınıflandırma işlemi yapılmıştır.

## 📁 Proje İçeriği

- **Veri seti:** [tic-tac-toe.data](https://archive.ics.uci.edu/ml/datasets/Tic-Tac-Toe+Endgame)
- **Model:** DecisionTreeClassifier (criterion="entropy")
- **Kod dosyası:** `KodDosyasi.py`
- **Rapor:** `BLM463_Proje_EmirCarikci_21360859060.pdf`
- **Görseller:** `confusion_matrix.png`, `decision_tree.png`, `roc_curve.png`

## 🔍 Kullanılan Kütüphaneler

- pandas
- scikit-learn
- matplotlib

## 📊 Değerlendirme Metrikleri

- Doğruluk (Accuracy): 89.58%
- Kesinlik (Precision): 94.54%
- Duyarlılık (Recall): 89.64%
- F1 Skoru: 92.02%
- ROC AUC: 0.8955

## 🔄 Çapraz Doğrulama

- 10-fold cross-validation kullanılarak modelin genellenebilirliği test edilmiştir.
- Ortalama doğruluk: %88.94
- Standart sapma: ±2.23

## 👨‍💻 Çalıştırma Talimatı

1. Gerekli kütüphaneleri yükleyin:
```bash
pip install pandas scikit-learn matplotlib
```

2. Python dosyasını çalıştırın:
```bash
python KodDosyasi.py
```

3. Çıktılar:
- confusion_matrix.png
- decision_tree.png
- roc_curve.png

## 📌 Öğrenci Bilgileri

- **Ad Soyad:** Emir Çarıkçı  
- **Öğrenci No:** 21360859060  
- **Üniversite:** Bursa Teknik Üniversitesi  
- **Ders:** BLM0463 - Veri Madenciliğine Giriş

