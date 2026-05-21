"""
IMAGINE Pipeline - Streamlit Frontend
Denoising Autoencoder + Stable Diffusion Image Generation

This application provides a user-friendly interface for the IMAGINE pipeline,
which takes noisy sketches, denoises them with a custom autoencoder, and then
generates full-color images using Stable Diffusion v1.5 img2img.
"""

import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import numpy as np
import os
import io
from streamlit_drawable_canvas import st_canvas


# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="IMAGINE Pipeline",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Theme-aware styling that works in both light and dark mode */
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: var(--text-color);
        margin-bottom: 0.25rem;
        border-bottom: 3px solid #4a90d9;
        padding-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: var(--text-color);
        opacity: 0.7;
        margin-bottom: 1.5rem;
    }
    .step-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--text-color);
        padding: 0.5rem 0;
        border-bottom: 2px solid rgba(128, 128, 128, 0.3);
        margin-bottom: 1rem;
    }
    .status-ok {
        color: #2ecc71;
        font-weight: 600;
    }
    .status-warn {
        color: #f39c12;
        font-weight: 600;
    }
    .status-error {
        color: #e74c3c;
        font-weight: 600;
    }
    .status-loading {
        color: #3498db;
        font-weight: 600;
    }
    .info-box {
        background-color: rgba(74, 144, 217, 0.1);
        border-left: 4px solid #4a90d9;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0 4px 4px 0;
        font-size: 0.9rem;
        color: var(--text-color);
    }
    .param-label {
        font-weight: 600;
        color: var(--text-color);
        font-size: 0.9rem;
    }
    .stButton > button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Model Architecture (exact copy from training notebook)
# ---------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    """Residual block with InstanceNorm and LeakyReLU."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.InstanceNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.InstanceNorm2d(out_channels)
        self.match = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        skip = self.match(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + skip)


class RobustUNet(nn.Module):
    """
    U-Net style denoising autoencoder.
    Input: [B, 1, 128, 128] grayscale
    Output: [B, 1, 128, 128] denoised grayscale
    """

    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = ResidualBlock(1, 64)
        self.enc2 = ResidualBlock(64, 128)
        self.enc3 = ResidualBlock(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout2d(0.2)
        # Bottleneck
        self.bottleneck = ResidualBlock(256, 512)
        # Decoder
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = ResidualBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = ResidualBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = ResidualBlock(128, 64)
        self.final = nn.Conv2d(64, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(self.dropout(e3)))
        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(self.dropout(d3))
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.sigmoid(self.final(d1))


# ---------------------------------------------------------------------------
# Device Detection
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "robust_autoencoder_best.pth"
SD_MODEL_ID = "runwayml/stable-diffusion-v1-5"


# ---------------------------------------------------------------------------
# Cached Model Loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def load_autoencoder():
    """Load the denoising autoencoder model. Cached across reruns."""
    if not os.path.exists(MODEL_PATH):
        return None, f"Model file '{MODEL_PATH}' not found. Place it in the app directory."

    try:
        model = RobustUNet().to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        param_count = sum(p.numel() for p in model.parameters())
        return model, f"Loaded ({param_count:,} parameters, {DEVICE.upper()})"
    except Exception as e:
        return None, f"Failed to load autoencoder: {str(e)}"


@st.cache_resource
def load_stable_diffusion():
    """Load Stable Diffusion v1.5 img2img pipeline. Cached across reruns."""
    try:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            SD_MODEL_ID,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
            token=st.secrets["HF_TOKEN"]
        )
        pipe = pipe.to(DEVICE)
        if DEVICE == "cuda":
            pipe.enable_attention_slicing()
        precision = "float16 (GPU)" if DEVICE == "cuda" else "float32 (CPU)"
        return pipe, f"Ready ({precision})"
    except Exception as e:
        return None, f"Failed to load: {str(e)}"


# ---------------------------------------------------------------------------
# Pipeline Functions
# ---------------------------------------------------------------------------
def denoise_image(model, pil_image):
    """
    Run the denoising autoencoder on an input image.

    Args:
        model: The loaded RobustUNet model
        pil_image: Input PIL image (any size, any mode)

    Returns:
        tuple: (noisy_display, denoised_display, denoised_rgb)
            - noisy_display: numpy array for display (128x128 grayscale)
            - denoised_display: numpy array for display (128x128 grayscale)
            - denoised_rgb: PIL Image (original size, RGB) for Stable Diffusion
    """
    # Convert to grayscale
    gray_image = pil_image.convert("L")
    orig_w, orig_h = gray_image.size

    # Resize to 128x128 and convert to tensor [0, 1]
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    noisy_tensor = transform(gray_image).unsqueeze(0).to(DEVICE)

    # Run autoencoder (inference only)
    with torch.no_grad():
        denoised_tensor = model(noisy_tensor)

    # Convert to numpy for display
    noisy_np = (noisy_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
    denoised_np = (denoised_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)

    # Upscale denoised back to original resolution for Stable Diffusion
    denoised_pil = Image.fromarray(denoised_np)
    denoised_orig = denoised_pil.resize((orig_w, orig_h), Image.LANCZOS)
    denoised_rgb = denoised_orig.convert("RGB")

    return noisy_np, denoised_np, denoised_rgb


def generate_image(pipe, denoised_rgb, prompt, strength, guidance, steps):
    """
    Run Stable Diffusion img2img on the denoised image.

    Args:
        pipe: Loaded StableDiffusionImg2ImgPipeline
        denoised_rgb: PIL Image (RGB) from denoising stage
        prompt: Text prompt for generation
        strength: How much to deviate from input (0.1 = faithful, 0.95 = creative)
        guidance: How strictly to follow prompt
        steps: Number of inference steps

    Returns:
        PIL Image: The generated output image
    """
    result = pipe(
        prompt=prompt,
        image=denoised_rgb,
        strength=strength,
        guidance_scale=guidance,
        num_inference_steps=steps,
    )
    return result.images[0]


def pil_to_bytes(pil_image, fmt="PNG"):
    """Convert a PIL image to bytes for download."""
    buffer = io.BytesIO()
    pil_image.save(buffer, format=fmt)
    buffer.seek(0)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# UI: Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="main-header">IMAGINE Pipeline</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Denoising Autoencoder + Stable Diffusion</div>',
        unsafe_allow_html=True,
    )

    # --- Device Info ---
    st.markdown("---")
    st.markdown('<p class="param-label">System</p>', unsafe_allow_html=True)
    if DEVICE == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        st.markdown(
            f'<div class="info-box"><span class="status-ok">GPU:</span> {gpu_name} ({vram:.1f} GB VRAM)</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="info-box"><span class="status-warn">CPU Mode</span> '
            "- Stable Diffusion will be slow. GPU recommended.</div>",
            unsafe_allow_html=True,
        )

    # --- Model Loading ---
    st.markdown("---")
    st.markdown('<p class="param-label">Models</p>', unsafe_allow_html=True)

    ae_model, ae_status = load_autoencoder()
    if ae_model is not None:
        st.markdown(
            f'<div class="info-box"><span class="status-ok">Autoencoder:</span> {ae_status}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="info-box"><span class="status-error">Autoencoder:</span> {ae_status}</div>',
            unsafe_allow_html=True,
        )

    # Stable Diffusion loading toggle
    load_sd = st.checkbox("Load Stable Diffusion model", value=False,
                          help="First load downloads ~4 GB. Check this when ready to generate.")

    sd_pipe = None
    sd_status = "Not loaded (check the box above to load)"
    if load_sd:
        with st.spinner("Loading Stable Diffusion v1.5 (this may take a moment)..."):
            sd_pipe, sd_status = load_stable_diffusion()

    if sd_pipe is not None:
        st.markdown(
            f'<div class="info-box"><span class="status-ok">Stable Diffusion:</span> {sd_status}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="info-box"><span class="status-loading">Stable Diffusion:</span> {sd_status}</div>',
            unsafe_allow_html=True,
        )

    # --- Prompt ---
    st.markdown("---")
    st.markdown('<p class="param-label">Prompt</p>', unsafe_allow_html=True)
    prompt = st.text_area(
        "Describe the image to generate",
        value="a photorealistic white dog with a red ribbon around its neck, vibrant colors, high detail",
        height=100,
        label_visibility="collapsed",
        help="Describe what you want the final image to look like. Be specific for better results.",
    )

    # --- Generation Parameters ---
    st.markdown("---")
    st.markdown('<p class="param-label">Generation Settings</p>', unsafe_allow_html=True)

    strength = st.slider(
        "Strength",
        min_value=0.1,
        max_value=0.95,
        value=0.65,
        step=0.05,
        help="How much to deviate from the denoised sketch. "
             "Lower = more faithful to sketch, Higher = more creative.",
    )

    guidance = st.slider(
        "Guidance Scale",
        min_value=1.0,
        max_value=20.0,
        value=8.0,
        step=0.5,
        help="How strictly the prompt is followed. 7-9 is typically a good range.",
    )

    steps = st.slider(
        "Inference Steps",
        min_value=10,
        max_value=50,
        value=30,
        step=5,
        help="Number of denoising steps. More steps = higher quality but slower.",
    )

    # --- Generate Button ---
    st.markdown("---")
    generate_btn = st.button(
        "Run Pipeline",
        type="primary",
        use_container_width=True,
        disabled=(ae_model is None),
    )


# ---------------------------------------------------------------------------
# UI: Main Area
# ---------------------------------------------------------------------------

# --- Step 1: Image Input ---
st.markdown('<div class="step-header">Step 1: Provide an Input Image</div>', unsafe_allow_html=True)
st.markdown(
    "Upload a noisy image or sketch, or draw one on the canvas below. "
    "The image will be converted to 128x128 grayscale for the denoising autoencoder."
)

input_method = st.radio(
    "Choose Input Method:",
    ["Upload Image", "Draw on Canvas"],
    horizontal=True,
    label_visibility="collapsed"
)

input_image = None

if input_method == "Upload Image":
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["png", "jpg", "jpeg", "bmp", "webp"],
        label_visibility="collapsed",
    )
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file)
        st.image(input_image, caption="Uploaded Image", width=300)

elif input_method == "Draw on Canvas":
    st.markdown(
        "Draw a sketch below. A **white background with black strokes** works best."
    )
    
    # CSS to ensure the canvas has a theme-accurate border
    # and a drop-shadow filter that outlines the black toolbar icons with the theme's text color.
    # This makes the undo/redo SVG icons pop out and become clearly visible in both Dark and Light modes!
    st.markdown("""
        <style>
        iframe[title="streamlit_drawable_canvas.st_canvas"] {
            border: 2px solid var(--text-color) !important;
            border-radius: 8px !important;
            filter: drop-shadow(0px 0px 3px var(--text-color));
        }
        </style>
    """, unsafe_allow_html=True)

    stroke_width = st.slider("Brush Size", min_value=1, max_value=50, value=15, step=1)
    
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 1)",
        stroke_width=stroke_width,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=400,
        width=400,
        drawing_mode="freedraw",
        key="drawing_canvas",
    )

    if canvas_result.image_data is not None:
        # The canvas result is RGBA, we use numpy to composite it over a white background
        canvas_rgba = canvas_result.image_data.astype(np.float32)
        alpha = (canvas_rgba[:, :, 3] / 255.0).reshape(400, 400, 1)
        rgb = canvas_rgba[:, :, :3]
        white_bg = np.ones((400, 400, 3), dtype=np.float32) * 255.0
        
        composited = (rgb * alpha) + (white_bg * (1.0 - alpha))
        composited = composited.astype(np.uint8)
        
        bg = Image.fromarray(composited, "RGB")
        
        # Check if the user has actually drawn something (look for dark pixels)
        gray_canvas = np.mean(composited, axis=2)
        non_white_pixels = np.sum(gray_canvas < 240)
        
        if non_white_pixels > 50:
            input_image = bg
            st.markdown(
                '<div class="info-box">Canvas sketch detected. Ready for processing.</div>',
                unsafe_allow_html=True,
            )

st.markdown("---")

# --- Step 2 & 3: Pipeline Execution ---
if generate_btn:
    if input_image is None:
        st.error("No input image provided. Please upload an image or draw a sketch first.")
    elif ae_model is None:
        st.error(
            f"Autoencoder model could not be loaded. {ae_status}"
        )
    elif not prompt.strip():
        st.error("Please enter a text prompt before generating.")
    else:
        # ---- Step 2: Denoising ----
        st.markdown(
            '<div class="step-header">Step 2: Denoising with Autoencoder</div>',
            unsafe_allow_html=True,
        )

        with st.spinner("Running autoencoder denoising..."):
            noisy_display, denoised_display, denoised_rgb = denoise_image(ae_model, input_image)

        col_noisy, col_denoised = st.columns(2)
        with col_noisy:
            st.image(noisy_display, caption="Input (128x128 Grayscale)", use_container_width=True)
        with col_denoised:
            st.image(denoised_display, caption="Denoised Output", use_container_width=True)

        st.success("Denoising complete.")

        st.markdown("---")

        # ---- Step 3: Stable Diffusion Generation ----
        st.markdown(
            '<div class="step-header">Step 3: Image Generation with Stable Diffusion</div>',
            unsafe_allow_html=True,
        )

        if sd_pipe is None:
            st.warning(
                "Stable Diffusion model is not loaded. "
                "Check 'Load Stable Diffusion model' in the sidebar to enable generation. "
                "Note: first load downloads ~4 GB."
            )
        else:
            st.markdown(
                f"**Prompt:** {prompt}  \n"
                f"**Strength:** {strength} | **Guidance:** {guidance} | **Steps:** {steps}"
            )
            if DEVICE == "cpu":
                st.info(
                    "Running on CPU. This may take several minutes. Please be patient."
                )

            with st.spinner("Generating image with Stable Diffusion..."):
                try:
                    generated_image = generate_image(
                        sd_pipe, denoised_rgb, prompt, strength, guidance, steps
                    )
                except Exception as e:
                    st.error(f"Generation failed: {str(e)}")
                    generated_image = None

            if generated_image is not None:
                # Show all three stages side by side
                st.markdown("### Results")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(noisy_display, caption="Input Sketch", use_container_width=True)
                with col2:
                    st.image(denoised_display, caption="Denoised", use_container_width=True)
                with col3:
                    st.image(generated_image, caption="Generated Output", use_container_width=True)

                st.success("Generation complete.")

                # Download button
                img_bytes = pil_to_bytes(generated_image)
                st.download_button(
                    label="Download Generated Image",
                    data=img_bytes,
                    file_name="imagine_output.png",
                    mime="image/png",
                )

elif input_image is None:
    st.markdown(
        '<div class="info-box">Provide an input image above, then click "Run Pipeline" in the sidebar.</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="info-box">Image ready. Configure the prompt and settings in the sidebar, '
        'then click "Run Pipeline".</div>',
        unsafe_allow_html=True,
    )