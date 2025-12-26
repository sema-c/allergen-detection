import gradio as gr
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# Model 
print("Model yükleniyor...")
model = YOLO('../models/best.pt')

# Alerjen bilgileri
ALLERGEN_INFO = {
    0: {'name': 'Süt Ürünleri', 'color': '#FFD700'},
    1: {'name': 'Gluten', 'color': '#FF8C00'},
    2: {'name': 'Yumurta', 'color': '#DC143C'},
    3: {'name': 'Deniz Ürünleri', 'color': '#4169E1'},
    4: {'name': 'Kuruyemiş', 'color': '#8B4513'},
}

def generate_heatmap(image, boxes, class_id):
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
    heatmap = np.zeros((h, w), dtype=np.float32)
    
    for box in boxes:
        if int(box.cls[0]) == class_id:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            heatmap[y1:y2, x1:x2] = np.maximum(heatmap[y1:y2, x1:x2], conf)
    
    if heatmap.max() == 0:
        return None
    
    heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
    
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    heatmap_colored = cv2.applyColorMap(
        (heatmap * 255).astype(np.uint8), 
        cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    img_resized = cv2.resize(img_array, (w, h))
    overlay = cv2.addWeighted(img_resized, 0.6, heatmap_colored, 0.4, 0)
    
    return overlay

def detect_and_show(image, show_heatmap=True):
    if image is None:
        return (gr.Row(visible=True), gr.Row(visible=False), None, "", 
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False))
    
    results = model.predict(source=image, conf=0.25, save=False, verbose=False)
    result = results[0]
    boxes = result.boxes
    
    result_img = result.plot()
    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    
    if len(boxes) == 0:
        rapor = "Alerjen tespit edilmedi."
        return (gr.Row(visible=False), gr.Row(visible=True), result_img, rapor,
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False))
    
    allergen_stats = {}
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        
        if cls_id not in allergen_stats:
            allergen_stats[cls_id] = []
        allergen_stats[cls_id].append(conf)
    
    rapor = "Tespit Edilen Allerjenler:\n\n"
    for cls_id, confidences in sorted(allergen_stats.items()):
        info = ALLERGEN_INFO[cls_id]
        avg_conf = np.mean(confidences)
        rapor += f"{info['name']}: %{avg_conf*100:.1f}\n"
    
    heatmaps = []
    
    if show_heatmap:
        for cls_id in sorted(allergen_stats.keys()):
            if len(heatmaps) >= 3:
                break
            heatmap = generate_heatmap(image, boxes, cls_id)
            if heatmap is not None:
                info = ALLERGEN_INFO[cls_id]
                heatmaps.append(gr.update(value=heatmap, visible=True, label=f"{info['name']}"))
    
    while len(heatmaps) < 3:
        heatmaps.append(gr.update(value=None, visible=False))
    
    return (gr.Row(visible=False), gr.Row(visible=True), result_img, rapor,
            heatmaps[0], heatmaps[1], heatmaps[2])

def reset_interface():
    return (gr.Row(visible=True), gr.Row(visible=False), None, "",
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False))

# GRADIO ARAYÜZÜ
with gr.Blocks(title="Alerjen Tespit Sistemi", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown('''
    # Alerjen Tespit Sistemi
    
    **Tespit edilebilen allerjenler:**
    Süt Ürünleri | Gluten | Yumurta | Deniz Ürünleri | Kuruyemiş
    ''')
    
    with gr.Row(visible=True) as upload_screen:
        with gr.Column():
            gr.Markdown("### Yemek Fotoğrafı Yükleyin")
            input_image = gr.Image(
                label="",
                type="pil",
                height=400,
                sources=["upload", "webcam"]
            )
            detect_btn = gr.Button(
                "Alerjen Tespiti Yap",
                variant="primary",
                size="lg"
            )
    
    with gr.Row(visible=False) as result_screen:
        with gr.Column(scale=2):
            output_image = gr.Image(
                label="Tespit Sonucu",
                height=450
            )
            output_text = gr.Textbox(
                label="",
                lines=6,
                max_lines=10
            )
            back_btn = gr.Button(
                "← Yeni Tespit Yap",
                variant="secondary",
                size="sm"
            )
        
        with gr.Column(scale=1):
            gr.Markdown("### Heat Maps")
            gr.Markdown("*Kırmızı = Yüksek aktivasyon*")
            
            with gr.Column():
                heatmap1 = gr.Image(label="", height=170, visible=True)
                heatmap2 = gr.Image(label="", height=170, visible=True)
                heatmap3 = gr.Image(label="", height=170, visible=True)
    
    gr.Markdown('''
    ---
    **Not:** Bu sistem araştırma amaçlıdır. Alerji durumlarında mutlaka uzmana danışın.
    ''')
    
    detect_btn.click(
        fn=detect_and_show,
        inputs=input_image,
        outputs=[upload_screen, result_screen, output_image, output_text, 
                 heatmap1, heatmap2, heatmap3]
    )
    
    back_btn.click(
        fn=reset_interface,
        inputs=None,
        outputs=[upload_screen, result_screen, output_image, output_text,
                 heatmap1, heatmap2, heatmap3]
    )

if __name__ == "__main__":
    print("\nArayüz başlatılıyor...")
    demo.launch(share=True, debug=False)
