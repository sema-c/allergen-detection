# Alerjen Tespit Sistemi

YOLOv8 kullanarak yemek görüntülerinden alerjen tespiti yapan derin öğrenme projesi.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Genel Bakış

Bu proje, yemek görüntülerindeki alerjenleri tespit eden bir nesne tanıma sistemi içermektedir. YOLOv8 modeli kullanılarak beş farklı alerjen kategorisi gerçek zamanlı olarak tespit edilebilmektedir.

### Tespit Edilen Allerjenler
- Süt Ürünleri
- Gluten
- Yumurta
- Deniz Ürünleri
- Kuruyemiş

## Model Performansı

### Genel Metrikler
- **Model**: YOLOv8n (Nano)
- **Görüntü Boyutu**: 416x416
- **Eğitim Epoch**: 50
- **Veri Seti**: UECFOOD-256 (15,805 görüntü)
- **Genel mAP@0.5**: 61.1%
- **Genel mAP@0.5-0.95**: 45.8%

### Sınıf Bazlı Performans

| Alerjen | mAP@0.5 | mAP@0.5-0.95 | Precision | Recall | Örnek Sayısı |
|---------|---------|--------------|-----------|--------|--------------|
| Gluten | 82.1% | 63.6% | 68.6% | 87.1% | 1,757 |
| Deniz Ürünleri | 69.2% | 47.6% | 62.6% | 74.8% | 654 |
| Süt Ürünleri | 66.0% | 51.7% | 55.6% | 78.4% | 533 |
| Yumurta | 53.9% | 40.0% | 49.6% | 69.4% | 589 |
| Kuruyemiş | 34.5% | 26.3% | 39.0% | 56.2% | 73 |

## Özellikler

- Gerçek zamanlı alerjen tespiti
- Çok sınıflı nesne tanıma (5 alerjen kategorisi)
- Web tabanlı arayüz (Gradio)
- Webcam desteği
- Isı haritası görselleştirme
- Toplu tahmin desteği
- Detaylı performans metrikleri

## Kurulum

### Gereksinimler
- Python 3.8 veya üzeri
- CUDA destekli GPU (önerilir)
- 8GB+ RAM

### Kurulum Adımları

1. Repository'yi klonlayın
```bash
git clone https://github.com/sema-c/allergen-detection.git
cd allergen-detection
```

2. Sanal ortam oluşturun (önerilir)
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
venv\Scripts\activate  # Windows
```

3. Bağımlılıkları yükleyin
```bash
pip install -r requirements.txt
```

4. Model dosyasını indirin

Eğitilmiş model dosyası (best.pt, ~6MB) boyut kısıtlaması nedeniyle ayrı olarak barındırılmaktadır:

**[Model İndir (best.pt - 6MB)](https://drive.google.com/file/d/1-sP7pEr0ZgFj2L0-25WVUluogDiqgM1g/view?usp=sharing)**

İndirilen dosyayı `models/` dizinine yerleştirin:
```bash
# Linux/Mac
mv ~/Downloads/best.pt models/

# Windows
move Downloads\best.pt models\
```

## Kullanım

### Web Arayüzü (Gradio)
```bash
python app/gradio_app.py
```
Arayüz otomatik olarak `http://localhost:7860` adresinde açılacaktır.

### Jupyter Notebook
```bash
jupyter notebook notebooks/training.ipynb
```

### Python API
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
├── notebooks/              # Jupyter notebook'lar
│   ├── training.ipynb     # Model eğitimi
│   └── evaluation.ipynb   # Model değerlendirme
├── app/                    # Web arayüzleri
│   ├── gradio_app.py      # Gradio arayüzü
│   └── streamlit_app.py   # Streamlit arayüzü
├── models/                 # Eğitilmiş modeller
│   └── best.pt            # En iyi model ağırlıkları
├── results/                # Eğitim sonuçları
│   ├── confusion_matrix.png
│   ├── results.png
│   └── training_curves.png
├── docs/                   # Dökümanlar
│   └── REPORT.md          # Detaylı proje raporu
├── assets/                 # Görseller ve medya
│   └── demo.gif           # Demo görselleştirme
├── requirements.txt        # Python bağımlılıkları
├── .gitignore             # Git ignore dosyası
├── LICENSE                # MIT Lisansı
└── README.md              # Bu dosya
```

## Metodoloji

### Veri Seti
- **Kaynak**: UECFOOD-256
- **Toplam Görüntü**: 31,397
- **Eğitim Seti**: 12,344 görüntü
- **Doğrulama Seti**: 3,461 görüntü
- **Etiketleme**: 18,097 alerjen etiketi

### Eğitim Parametreleri
- **Model**: YOLOv8n
- **Optimizer**: AdamW
- **Öğrenme Oranı**: 0.01
- **Batch Boyutu**: 32
- **Epoch**: 50
- **Görüntü Boyutu**: 416x416
- **Veri Artırma**: Mosaic, MixUp, HSV, Flip

### Veri Artırma Teknikleri
- Rastgele yatay çevirme
- HSV renk değişimi
- Mosaic artırma
- MixUp artırma

## Sonuçlar

### Güçlü Yönler
- Gluten tespitinde en yüksek performans: 82.1% mAP@0.5
- Ortalama 73.2% recall oranı ile güçlü tespit kabiliyeti
- Deniz ürünleri ve süt ürünlerinde dengeli performans

### Sınırlamalar
- Kuruyemiş sınıfında düşük performans (sadece 73 eğitim örneği)
- Yumurta tespitinde orta seviye doğruluk
- Küçük nesnelerin tespitinde zorluklar

## Alıntı

Bu projeyi araştırmanızda kullanıyorsanız, lütfen alıntı yapın:
```bibtex
@misc{allergen_detection_2024,
  title={YOLOv8 ile Alerjen Tespit Sistemi},
  author={TÜBİTAK 2209-A Araştırma Projesi},
  year={2024},
  howpublished={\url{https://github.com/sema-c/allergen-detection}}
}
```

## Sorumluluk Reddi

**ÖNEMLİ**: Bu sistem yalnızca araştırma amaçlıdır ve tıbbi karar verme için kullanılmamalıdır. Gıda alerjileri ve diyet kısıtlamaları konusunda mutlaka sağlık uzmanlarına danışın. Model yüzde 100 doğruluk garantisi vermez ve kritik uygulamalarda kullanılmadan önce kapsamlı testlerden geçirilmelidir.

## Katkıda Bulunma

Katkılarınızı bekliyoruz! Lütfen şu adımları izleyin:

1. Repository'yi fork edin
2. Yeni bir özellik dalı oluşturun (`git checkout -b feature/iyilestirme`)
3. Değişikliklerinizi commit edin (`git commit -m 'Yeni özellik eklendi'`)
4. Dalınızı push edin (`git push origin feature/iyilestirme`)
5. Pull Request açın

## Lisans

Bu proje MIT Lisansı altında lisanslanmıştır - detaylar için [LICENSE](LICENSE) dosyasına bakın.

## Teşekkürler

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - Nesne tanıma framework'ü
- [UECFOOD-256 Dataset](http://foodcam.mobi/dataset256.html) - Veri seti sağlayıcısı
- [TÜBİTAK 2209-A Programı](https://www.tubitak.gov.tr) - Araştırma desteği
- [Gradio](https://gradio.app/) - Web arayüz framework'ü

## İletişim

Sorularınız veya sorunlarınız için lütfen GitHub üzerinden issue açın.

---

Bu projeyi faydalı bulduysanız yıldız vermeyi unutmayın!
