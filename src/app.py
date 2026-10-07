import numpy as np
import torch
import gradio as gr

from model import SmallUNet
from inference import enhance_full_image

CHECKPOINT_PATH = "../checkpoints/best.pth"
device = torch.device("cpu")

model = SmallUNet().to(device)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.eval()
print("Model loaded — app is ready")


def enhance(image):
    if image is None:
        return None
    img = image.astype(np.float32)
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    else:
        img = np.zeros_like(img)
    enhanced = enhance_full_image(model, img)
    enhanced = np.clip(enhanced, 0, 1)
    return (enhanced * 255).astype(np.uint8)


# All custom visual styling lives here, as plain CSS, and is attached via elem_id
custom_css = """
.gradio-container {
    background-color: #000000 !important;
}

#header-box {
    background: linear-gradient(135deg, #111111 0%, #1a1a1a 100%);
    border: 1px solid #2a2a2a;
    border-radius: 12px;
    padding: 24px 30px;
    margin-bottom: 20px;
    width: 100%;
    box-shadow: 0 4px 20px rgba(0, 200, 180, 0.08);
}

#header-box .header-content {
    display: flex;
    align-items: center;
    gap: 20px;
}

#header-box .logo {
    font-size: 3em;
    flex-shrink: 0;
}

#header-box .title-block {
    text-align: left;
}

#header-box h1 {
    font-family: 'Times New Roman', Times, serif !important;
    color: #ffffff;
    margin: 0;
    font-size: 2.2em;
}

#header-box .subtitle {
    font-family: 'Times New Roman', Times, serif !important;
    color: #5fd4c4;
    font-size: 1em;
    margin-top: 6px;
    letter-spacing: 1px;
}

#header-box .accent-line {
    width: 80px;
    height: 3px;
    background: linear-gradient(90deg, #5fd4c4, #3a9d8f);
    margin-top: 14px;
    border-radius: 2px;
}

#description-box, #rules-box {
    background-color: #111111;
    border: 1px solid #2a2a2a;
    border-radius: 12px;
    padding: 22px;
    margin-bottom: 20px;
    width: 100%;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
}

#description-box p, #description-box li,
#rules-box p, #rules-box li {
    font-family: 'Times New Roman', Times, serif !important;
    color: #eeeeee;
    font-size: 16px;
    line-height: 1.6;
}

#description-box strong, #rules-box strong {
    color: #5fd4c4;
}
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Base()) as demo:

    # 1. Header — logo top-left, title/subtitle/accent-line beside it
    with gr.Column(elem_id="header-box"):
        gr.HTML(
            """
            <div class="header-content">
                <div class="logo">🔬</div>
                <div class="title-block">
                    <h1>Microscope Image Enhancer</h1>
                    <div class="subtitle">Self-Supervised Denoising for Metaphase &amp; Anaphase Imaging</div>
                    <div class="accent-line"></div>
                </div>
            </div>
            """
        )

    # 2. Description box
    with gr.Column(elem_id="description-box"):
        gr.Markdown(
            """
            **🧬 What this tool does**

            This tool cleans up noisy or blurry microscope images taken during
            metaphase and anaphase. It was trained using a self-supervised method —
            meaning it learned to reduce noise directly from real microscope images,
            without ever needing a separate "clean" reference image to learn from.
            """
        )

    # 3. Rules / how it works box
    with gr.Column(elem_id="rules-box"):
        gr.Markdown(
            """
            **⚙️ How it works**

            1. 📤 Upload a grayscale microscope image — metaphase or anaphase
               chromosome spread images work best, since that's what the model
               was trained on.
            2. 🧩 The tool breaks the image into small overlapping pieces, cleans
               each piece individually, and blends them back together smoothly.
            3. 📐 Any image size is supported, not just one fixed size.
            4. ✨ The enhanced result appears below automatically once uploaded.
            """
        )

    # 4 & 5. Upload and result boxes, side by side
    with gr.Row():
        image_input = gr.Image(type="numpy", image_mode="L", label="📤 Upload microscope image")
        image_output = gr.Image(type="numpy", label="✨ Enhanced result")

    image_input.upload(fn=enhance, inputs=image_input, outputs=image_output)

if __name__ == "__main__":
    demo.launch()