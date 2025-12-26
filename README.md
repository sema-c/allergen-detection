# Alerjen Tespit Sistemi

YOLOv8 kullanarak yemek görüntülerinden alerjen tespiti yapan derin öğrenme projesi.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Proje Özeti

Bu proje bir derin öğrenme uygulamasıdır. YOLOv8 object detection modeli kullanılarak yemek görüntülerinden 5 farklı alerjen sınıfı tespit edilmektedir.

### Tespit Edilen Allerjenler
- 🟡 Süt Ürünleri (Dairy)
- 🟠 Gluten
- 🔴 Yumurta (Egg)
- 🔵 Deniz Ürünleri (Seafood)
- 🟤 Kuruyemiş (Nuts)

## Model Performansı

### Genel Metrikler
- **Model**: YOLOv8n (Nano)
- **Image Size**: 416x416
- **Epochs**: 50
- **Dataset**: UECFOOD-256 (15,805 görüntü)
- **Overall mAP@0.5**: 61.1%
- **Overall mAP@0.5-0.95**: 45.8%

### Sınıf Bazlı Performans

| Allerjen | mAP@0.5 | mAP@0.5-0.95 | Precision | Recall | Örnekler |
|----------|---------|--------------|-----------|--------|----------|
| **Gluten** | 82.1% | 63.6% | 68.6% | 87.1% | 1,757 |
| **Deniz Ürünleri** | 69.2% | 47.6% | 62.6% | 74.8% | 654 |
| **Süt Ürünleri** | 66.0% | 51.7% | 55.6% | 78.4% | 533 |
| **Yumurta** | 53.9% | 40.0% | 49.6% | 69.4% | 589 |
| **Kuruyemiş** | 34.5% | 26.3% | 39.0% | 56.2% | 73 |

## Özellikler

- Real-time alerjen tespiti
- 5 farklı alerjen sınıfı
- Web tabanlı arayüzler (Gradio)
- Webcam desteği
- Heat map görselleştirme
- Batch prediction desteği
- Detaylı performans metrikleri

## Kurulum

### Gereksinimler
- Python 3.8 veya üzeri
- CUDA destekli GPU (önerilir)
- 8GB+ RAM

### 1. Repository'yi Klonlayın
```bash
git clone https://github.com/KULLANICI_ADINIZ/allergen-detection.git
cd allergen-detection
```

### 2. Sanal Ortam Oluşturun (Önerilir)
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
venv\Scripts\activate  # Windows
```

### 3. Gereksinimleri Yükleyin
```bash
pip install -r requirements.txt
```

### 4. Model Dosyasını İndirin
Model dosyası (best.pt) boyutu nedeniyle Git LFS ile yönetilmektedir veya aşağıdaki linkten indirebilirsiniz:
- [Google Drive Linki](https://drive.google.com/LINK_BURAYA_GELECEK)

İndirdikten sonra `models/` klasörüne yerleştirin.

## Kullanım

### Gradio Web Arayüzü
```bash
python app/gradio_app.py
```
Tarayıcınızda otomatik olarak `http://localhost:7860` adresinde açılacaktır.


### Jupyter Notebook ile Eğitim
```bash
jupyter notebook notebooks/training.ipynb
```

### Python Script ile Tahmin
```python
from ultralytics import YOLO

# Model yükle
model = YOLO('models/best.pt')

# Tahmin yap
results = model.predict('path/to/image.jpg', conf=0.25)

# Sonuçları göster
results[0].show()
```

## Proje Yapısı
```
allergen-detection/
│
├── notebooks/              # Jupyter notebook'lar
│   ├── training.ipynb     # Model eğitim notebook'u
│   └── evaluation.ipynb   # Model değerlendirme notebook'u
│
├── app/                    # Web arayüzleri
│   ├── gradio_app.py      # Gradio arayüzü
│   └── streamlit_app.py   # Streamlit arayüzü
│
├── models/                 # Eğitilmiş modeller
│   └── best.pt            # En iyi model (Git LFS)
│
├── results/                # Eğitim sonuçları
│   ├── confusion_matrix.png
│   ├── results.png
│   └── training_curves.png
│
├── docs/                   # Dökümanlar
│   └── REPORT.md          # Detaylı proje raporu
│
├── assets/                 # Görseller
│   └── demo.gif           # Demo görseli
│
├── requirements.txt        # Python bağımlılıkları
├── .gitignore             # Git ignore dosyası
├── LICENSE                # Lisans
└── README.md              # Bu dosya
```

## Metodoloji

### Dataset
- **Kaynak**: UECFOOD-256
- **Toplam Görüntü**: 31,397
- **Eğitim Seti**: 12,344 görüntü
- **Validation Seti**: 3,461 görüntü
- **Annotasyon**: 18,097 alerjen etiketi

### Eğitim Parametreleri
- **Model**: YOLOv8n
- **Optimizer**: AdamW
- **Learning Rate**: 0.01
- **Batch Size**: 32
- **Epochs**: 50
- **Image Size**: 416x416
- **Augmentation**: Mosaic, MixUp, HSV, Flip

### Veri Artırma
- Random horizontal flip
- HSV color jittering
- Mosaic augmentation
- MixUp augmentation

## Sonuçlar

### Başarılı Tespitler
- Gluten tespitinde %82.1 mAP@0.5 ile en yüksek performans
- Ortalama recall %73.2 - modelin çoğu alerjeni yakaladığını gösterir
- Deniz ürünleri ve süt ürünlerinde dengeli performans

### Zorluklar
- Kuruyemiş sınıfında düşük performans (sadece 73 örnek)
- Yumurta tespitinde orta seviye başarı
- Küçük objelerde tespit zorluğu

## Akademik Kullanım

Bu proje akademik amaçlı geliştirilmiştir. Kullanım ve atıf için:
```bibtex
@misc{allergen_detection_2024,
  title={YOLOv8 ile Alerjen Tespit Sistemi},
  author={Sema},
  year={2024},
  note={TÜBİTAK 2209-A Araştırma Projesi}
}
```

## Önemli Uyarılar

- Bu sistem **araştırma amaçlıdır** ve tıbbi karar vermek için kullanılmamalıdır
- Alerji durumlarında **mutlaka uzmana danışın**
- Model %100 doğruluk sağlamaz, hata payı vardır
- Kritik uygulamalarda kullanmadan önce kapsamlı test yapılmalıdır

## Katkıda Bulunma

Katkılarınızı bekliyoruz! Lütfen:
1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit yapın (`git commit -m 'feat: Add amazing feature'`)
4. Push edin (`git push origin feature/amazing-feature`)
5. Pull Request açın

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - Object detection framework
- [UECFOOD-256 Dataset](http://foodcam.mobi/dataset256.html) - Dataset sağlayıcısı
- [TÜBİTAK 2209-A Programı](https://www.tubitak.gov.tr/tr/burslar/lisans/burs-programlari/2209-a/icerik-2209-universite-ogrencileri-arastirma-projeleri-destekleme-programi) - Proje desteği
- [Gradio](https://gradio.app/) & [Streamlit](https://streamlit.io/) - Web arayüz araçları

