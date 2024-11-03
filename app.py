import cv2
import numpy as np
import gradio as gr
from datetime import datetime

def apply_gaussian_blur(frame):
    return cv2.GaussianBlur(frame, (15, 15), 0)

def apply_sharpening_filter(frame):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(frame, -1, kernel)

def apply_edge_detection(frame):
    return cv2.Canny(frame, 100, 200)

def apply_invert_filter(frame):
    return cv2.bitwise_not(frame)

def adjust_brightness_contrast(frame, alpha=1.0, beta=50):
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

def apply_grayscale_filter(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def apply_sepia_filter(frame):
    sepia_filter = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
    return cv2.transform(frame, sepia_filter)

def apply_fall_filter(frame):
    fall_filter = np.array([[0.393, 0.769, 0.189],
                            [0.349, 0.686, 0.168],
                            [0.272, 0.534, 0.131]])
    return cv2.transform(frame, fall_filter)

def apply_bilateral_filter(frame):
    return cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)

def apply_histogram_equalization(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gray_frame)

def apply_crop(frame, x_start, y_start, width, height):
    return frame[y_start:y_start + height, x_start:x_start + width]

def apply_upscale(frame):
    return cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

def apply_rotation(frame, angle):
    h, w = frame.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(frame, M, (w, h))

def apply_reflection(frame):
    return cv2.flip(frame, 1)

def create_collage(images, rows, cols):
    if len(images) == 0:
        return None
    image_height, image_width = images[0].shape[:2]
    collage = np.zeros((rows * image_height, cols * image_width, 3), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            if (i * cols + j) < len(images):
                collage[i * image_height:(i + 1) * image_height,
                        j * image_width:(j + 1) * image_width] = images[i * cols + j]
    return collage

def apply_smoothing(frame):
    return cv2.GaussianBlur(frame, (5, 5), 0)

def apply_masking(frame, mask):
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return cv2.bitwise_and(frame, mask)

def apply_effect(frame, effect_type):
    if effect_type == "Cartoon":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(frame, 9, 300, 300)
        return cv2.bitwise_and(color, color, mask=edges)

    elif effect_type == "Sepia":
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        return cv2.transform(frame, kernel)

    elif effect_type == "Negative":
        return cv2.bitwise_not(frame)

    elif effect_type == "Emboss":
        kernel = np.array([[2, 0, 0],
                           [0, -1, 0],
                           [0, 0, -1]])
        return cv2.filter2D(frame, -1, kernel)

    elif effect_type == "Pixelate":
        height, width = frame.shape[:2]
        small = cv2.resize(frame, (width // 10, height // 10), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)

    return frame

def record_video(duration):
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi", fourcc, 20.0, (640, 480))
    
    start_time = datetime.now()
    while (datetime.now() - start_time).seconds < duration:
        ret, frame = cap.read()
        if ret:
            out.write(frame)
            cv2.imshow('Video Kaydı', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def save_image(image):
    filename = f"saved_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    cv2.imwrite(filename, image)
    return f"Resim kaydedildi: {filename}"

def apply_filter(filter_type, input_image=None, x_start=None, y_start=None, width=None, height=None, alpha=None, beta=None, angle=None, effect_type=None):
    if input_image is not None:
        frame = input_image
    else:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return "Web kameradan görüntü alınamadı"

    if filter_type == "Gaussian Blur":
        return apply_gaussian_blur(frame)
    elif filter_type == "Sharpen":
        return apply_sharpening_filter(frame)
    elif filter_type == "Edge Detection":
        return apply_edge_detection(frame)
    elif filter_type == "Invert":
        return apply_invert_filter(frame)
    elif filter_type == "Brightness":
        return adjust_brightness_contrast(frame, alpha=alpha, beta=beta)
    elif filter_type == "Grayscale":
        return apply_grayscale_filter(frame)
    elif filter_type == "Sepia":
        return apply_sepia_filter(frame)
    elif filter_type == "Sonbahar":
        return apply_fall_filter(frame)
    elif filter_type == "Bilateral":
        return apply_bilateral_filter(frame)
    elif filter_type == "Histogram Equalization":
        return apply_histogram_equalization(frame)
    elif filter_type == "Crop":
        return apply_crop(frame, x_start, y_start, width, height)
    elif filter_type == "Upscale":
        return apply_upscale(frame)
    elif filter_type == "Rotate":
        return apply_rotation(frame, angle)
    elif filter_type == "Reflection":
        return apply_reflection(frame)
    elif filter_type == "Collage":
        return create_collage([frame], 1, 1) 
    elif filter_type == "Smoothing":
        return apply_smoothing(frame)
    elif filter_type == "Masking":
        return apply_masking(frame, frame)  
    elif filter_type == "Effect":
        return apply_effect(frame, effect_type)

def reset_image():
    return None

with gr.Blocks() as demo:
    gr.Markdown("# Filtre Uygulaması")
    gr.Markdown("Bu uygulama ile farklı filtreler ve efektler uygulayarak görüntüleri düzenleyebilirsiniz.")

    with gr.Row():
        input_image = gr.Image(label="Resim Yükle", type="numpy")

        filter_type = gr.Dropdown(
            label="Filtre Seçin",
            choices=["Gaussian Blur", "Sharpen", "Edge Detection", "Invert", "Brightness", "Grayscale", "Sepia", "Sonbahar", "Bilateral", "Histogram Equalization", "Crop", "Upscale", "Rotate", "Reflection", "Collage", "Smoothing", "Masking", "Effect"],
            value="Gaussian Blur"
        )

        effect_type = gr.Dropdown(label="Efekt Seçin", choices=["Cartoon", "Sepia", "Negative", "Emboss", "Pixelate"], value="Cartoon")

        x_start = gr.Slider(minimum=0, maximum=640, label="Başlangıç X Koordinatı", step=1)
        y_start = gr.Slider(minimum=0, maximum=480, label="Başlangıç Y Koordinatı", step=1)
        width = gr.Slider(minimum=1, maximum=640, label="Kırpılacak Genişlik", step=1)
        height = gr.Slider(minimum=1, maximum=480, label="Kırpılacak Yükseklik", step=1)

        alpha = gr.Slider(minimum=0.0, maximum=3.0, label="Parlaklık Katsayısı (alpha)", step=0.1, value=1.0)
        beta = gr.Slider(minimum=-100, maximum=100, label="Parlaklık Sabiti (beta)", step=1, value=50)

        angle = gr.Slider(minimum=0, maximum=360, label="Döndürme Açısı (degrees)", step=1, value=0)

        

    duration = gr.Number(label="Kayıt Süresi (saniye)", value=5)
    record_button = gr.Button("Video Kaydet")

    with gr.Row():
        apply_button = gr.Button("Filtreyi Uygula")
        save_button = gr.Button("Kaydet")
        reset_button = gr.Button("Geri Al")
        clear_button = gr.Button("Sil")

    output_image = gr.Image(label="Filtre Uygulandı")
    output_file = gr.File(label="Kaydedilen Görüntü")

    apply_button.click(fn=apply_filter, inputs=[filter_type, input_image, x_start, y_start, width, height, alpha, beta, angle, effect_type], outputs=output_image)

    record_button.click(fn=record_video, inputs=duration, outputs=None)

    save_button.click(fn=save_image, inputs=output_image, outputs=output_image)

    reset_button.click(fn=reset_image, inputs=None, outputs=output_image)

    clear_button.click(fn=reset_image, inputs=None, outputs=input_image)

    gr.Markdown("### Uygulama Hakkında")
    gr.Markdown("Bu uygulama, görsel içeriklerinizi daha ilginç hale getirmenize yardımcı olmak için bir dizi filtre ve efekt sunar.")

demo.launch()


