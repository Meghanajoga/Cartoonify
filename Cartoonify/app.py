import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Set page config
st.set_page_config(page_title="Cartoonify", page_icon="🎨")

st.title("🎨 Cartoonify — Image Processing Demo")

# Sidebar sliders for parameters
st.sidebar.header("Cartoonify Parameters")
num_bilateral = st.sidebar.slider("Number of Bilateral Filters", 1, 20, 7)
d = st.sidebar.slider("Diameter (d)", 1, 20, 9)
sigma_color = st.sidebar.slider("Sigma Color", 1, 200, 75)
sigma_space = st.sidebar.slider("Sigma Space", 1, 200, 75)
median_blur_ksize = st.sidebar.slider("Median Blur Ksize (odd)", 1, 31, 7)
edge_block_size = st.sidebar.slider("Edge Block Size (odd)", 3, 31, 9)
edge_C = st.sidebar.slider("Edge Threshold C", 1, 10, 2)

# Ensure odd sizes for median blur and edge block
if median_blur_ksize % 2 == 0:
    median_blur_ksize += 1
if edge_block_size % 2 == 0:
    edge_block_size += 1

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])

# Cartoonify function
def cartoonify_image(img_bgr, num_bilateral, d, sigma_color, sigma_space, median_blur_ksize, edge_block_size, edge_C):
    # Resize if too large
    h, w = img_bgr.shape[:2]
    max_dim = 1200
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

    # Apply bilateral filter
    img_color = img_bgr.copy()
    for _ in range(max(1, int(num_bilateral))):
        img_color = cv2.bilateralFilter(img_color, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)

    # Convert to grayscale and blur
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.medianBlur(img_gray, median_blur_ksize)

    # Detect edges
    edges = cv2.adaptiveThreshold(img_blur, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY,
                                  edge_block_size,
                                  edge_C)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cartoon = cv2.bitwise_and(img_color, edges_colored)
    return cartoon

# Process uploaded file
if uploaded_file is not None:
    try:
        # Try OpenCV first
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # If OpenCV fails (None), fallback to PIL
        if img_bgr is None:
            uploaded_file.seek(0)  # reset file pointer
            pil_img = Image.open(uploaded_file).convert("RGB")
            img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # Only process if image loaded correctly
        if img_bgr is not None:
            cartoon_img = cartoonify_image(
                img_bgr, num_bilateral, d, sigma_color, sigma_space,
                median_blur_ksize, edge_block_size, edge_C
            )

            st.subheader("Original Image")
            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

            st.subheader("Cartoonified Image")
            st.image(cv2.cvtColor(cartoon_img, cv2.COLOR_BGR2RGB), use_container_width=True)

    except Exception as e:
        st.error(f"Could not process the uploaded image. Error: {e}")



