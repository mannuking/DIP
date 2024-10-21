import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from skimage.metrics import peak_signal_noise_ratio as psnr
import random
import matplotlib.pyplot as plt
import utils  # Import your utils.py file

# --- Initialize Session State ---
if "page" not in st.session_state:
    st.session_state.page = 0
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

# --- Functions to Manage Pages ---
def next_page():
    st.session_state.page += 1


def prev_page():
    st.session_state.page -= 1


def display_page(page_number):
    if page_number == 0:
        page_upload()
    elif page_number == 1:
        page_bit_depth()
    elif page_number == 2:
        page_filtering()
    elif page_number == 3:
        page_noise()
    elif page_number == 4:
        page_compression()
    elif page_number == 5:
        page_color_conversion()
    elif page_number == 6:
        page_histogram_equalization()
    elif page_number == 7:
        page_segmentation()
    elif page_number == 8:
        page_fourier_transform()


# --- Page Content Functions ---
def page_upload():
    st.title("Digital Image Processing Interactive Demo 📸")
    st.header("Upload Image ⬆️")
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_image is not None:
        st.session_state.uploaded_image = Image.open(uploaded_image)
        st.image(st.session_state.uploaded_image, caption="Uploaded Image", use_column_width=True)
        st.button("Next", on_click=next_page)


def page_bit_depth():
    st.header("Bit Depth Adjustment 🖼️")
    if st.session_state.uploaded_image is not None:
        img_array = np.array(st.session_state.uploaded_image)
        bit_depth = st.slider("Bit Depth (1-8 bits)", 1, 8, 8)
        modified_img = utils.reduce_bit_depth(img_array, bit_depth)
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_array, caption="Original Image", use_column_width=True)
        with col2:
            st.image(modified_img, caption=f"Image with {bit_depth}-bit depth", use_column_width=True)

        st.markdown("""
        **🤔 What is Bit Depth?**
        Bit depth refers to the number of bits used to represent each pixel in an image. A higher bit depth allows for a greater range of colors or shades of gray. 

        **💡 Analogy:** Think of a paint set. A set with only 8 colors (low bit depth) will limit your artistic expression, while a set with 256 colors (higher bit depth) offers more possibilities.

        **🧮 Formula:**
        The number of colors (or gray levels) that can be represented by a given bit depth is calculated as: 
        
        $$
        \text{Number of Colors} = 2^{\text{Bit Depth}}
        $$ 

        For example, an 8-bit image can represent $2^8 = 256$ colors. 
        """)

        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            st.button("Previous", on_click=prev_page)
        with col3:
            st.button("Next", on_click=next_page)
    else:
        st.write("Please upload an image on the previous page.")
        st.button("Previous", on_click=prev_page)


def page_filtering():
    st.header("Image Filtering 🧹")
    if st.session_state.uploaded_image is not None:
        img_array = np.array(st.session_state.uploaded_image)
        filter_type = st.selectbox(
            "Select a filter:",
            ["Gaussian Blur", "Sharpen", "Sobel Edge Detection", "Canny Edge Detection"],
        )

        if filter_type == "Gaussian Blur":
            kernel_size = st.slider("Kernel Size", 3, 15, 5, step=2)
            filtered_img = utils.gaussian_blur(img_array, kernel_size)
            explanation = """
            **Gaussian Blur:** Smooths the image by averaging pixel values within a kernel, reducing noise and detail. 
            The kernel size determines the extent of blurring.

            **💡 Analogy:** Imagine rubbing a blurry piece of glass over the image. The larger the glass, the more blurred the image becomes.
            """
        elif filter_type == "Sharpen":
            intensity = st.slider("Sharpening Intensity", 1.0, 5.0, 1.5)
            filtered_img = utils.sharpen(img_array, intensity)
            explanation = """
            **Sharpen:** Enhances edges and details in the image by increasing the contrast between neighboring pixels. 
            The sharpening intensity controls the strength of the effect.

            **💡 Analogy:** Think of tracing the outlines of objects in the image with a darker pen, making them stand out more.
            """
        elif filter_type == "Sobel Edge Detection":
            filtered_img = utils.sobel_edge_detection(utils.grayscale(img_array))
            explanation = """
            **Sobel Edge Detection:** Detects edges in the image by calculating gradients in horizontal and vertical directions. 
            It highlights regions with rapid changes in pixel intensity.

            **💡 Analogy:** Imagine an artist sketching the image, focusing only on the outlines and boundaries of objects.
            """
        elif filter_type == "Canny Edge Detection":
            low_threshold = st.slider("Low Threshold", 0, 255, 50)
            high_threshold = st.slider("High Threshold", 0, 255, 150)
            filtered_img = utils.canny_edge_detection(
                utils.grayscale(img_array), low_threshold, high_threshold
            )
            explanation = """
            **Canny Edge Detection:** A multi-step edge detection algorithm that identifies edges by suppressing noise, 
            finding gradients, and applying hysteresis thresholding.

            **💡 Analogy:** Think of a sophisticated image editing software automatically selecting the most prominent edges in the image, 
            like outlining important features for a graphic design.
            """

        col1, col2 = st.columns(2)
        with col1:
            st.image(img_array, caption="Original Image", use_column_width=True)
        with col2:
            st.image(filtered_img, caption=f"Filtered Image ({filter_type})", use_column_width=True)

        # Explanation after the images
        st.markdown(explanation) 

        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            st.button("Previous", on_click=prev_page)
        with col3:
            st.button("Next", on_click=next_page)
    else:
        st.write("Please upload an image on the first page.")
        st.button("Previous", on_click=prev_page)


def page_noise():
    st.header("4. Noise Addition and Removal 🧂")
    if st.session_state.uploaded_image is not None:
        img_array = np.array(st.session_state.uploaded_image)
        noise_type = st.selectbox("Select Noise Type:", ["Gaussian Noise", "Salt & Pepper Noise"])

        if noise_type == "Gaussian Noise":
            mean = st.slider("Mean (Noise Level)", 0.0, 1.0, 0.05)
            var = st.slider("Variance", 0.0, 0.1, 0.01)
            noisy_img = utils.add_gaussian_noise(img_array, mean, var)
            explanation = """
            **Gaussian Noise:**  Adds random noise following a Gaussian (normal) distribution. 
            The mean controls the overall noise level, and the variance determines the spread of the noise. 
            """
        elif noise_type == "Salt & Pepper Noise":
            prob = st.slider("Noise Probability", 0.0, 0.1, 0.01)
            noisy_img = utils.add_salt_and_pepper_noise(img_array, prob)
            explanation = """
            **Salt & Pepper Noise:**  Replaces random pixels with either black (pepper) or white (salt) values. 
            The noise probability determines how many pixels are affected.
            """

        filter_type_noise = st.selectbox("Select a Restoration Filter:", ["Mean Filter", "Median Filter"])
        if filter_type_noise == "Mean Filter":
            kernel_size = st.slider("Kernel Size", 3, 15, 3, step=2)
            restored_img = utils.mean_filter(noisy_img, kernel_size)
            filter_explanation = """
            **Mean Filter:** Averages pixel values within a kernel, effectively smoothing out noise.
            Larger kernel sizes result in stronger smoothing but may blur details.
            """
        elif filter_type_noise == "Median Filter":
            kernel_size = st.slider("Kernel Size", 3, 15, 3, step=2)
            restored_img = utils.median_filter(noisy_img, kernel_size)
            filter_explanation = """
            **Median Filter:** Replaces the center pixel with the median value within the kernel. 
            It's effective at removing impulsive noise (like salt & pepper) while preserving edges.
            """

        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(img_array, caption="Original Image", use_column_width=True)
        with col2:
            st.image(noisy_img, caption=f"Noisy Image ({noise_type})", use_column_width=True, clamp=True)
        with col3:
            st.image(restored_img, caption=f"Restored Image ({filter_type_noise})", use_column_width=True, clamp=True)

        st.markdown(explanation) # Noise explanation
        st.markdown(filter_explanation) # Filter explanation

        st.markdown("""
        **💡 Analogy:** Imagine a photo that has been damaged (noise). Restoration filters are like tools used to repair the photo. 
        Some tools might blend colors (mean filter) to smooth out scratches, 
        while others might replace damaged spots with colors from nearby areas (median filter). 
        """)

        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            st.button("Previous", on_click=prev_page)
        with col3:
            st.button("Next", on_click=next_page)
    else:
        st.write("Please upload an image on the first page.")
        st.button("Previous", on_click=prev_page)


def page_compression():
    st.header("Image Compression and Quality 🗜️")
    if st.session_state.uploaded_image is not None:
        img_array = np.array(st.session_state.uploaded_image)

        # Convert the original image to RGB if it has an alpha channel
        if img_array.shape[2] == 4:  # Check for alpha channel
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

        compression_type = st.selectbox("Select a compression type:", ["JPEG", "PNG"])

        if compression_type == "JPEG":
            quality = st.slider("JPEG Quality (1-100)", 1, 100, 75)
            compressed_img = utils.jpeg_compression(img_array, quality)

            # Ensure the compressed image is also RGB
            if compressed_img.shape[2] == 4:
                compressed_img = cv2.cvtColor(compressed_img, cv2.COLOR_RGBA2RGB)

            _, encoded_img = cv2.imencode(
                ".jpg", compressed_img, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            )
            file_size = len(encoded_img) / 1024
            psnr_value = psnr(img_array, compressed_img)
            caption = f"Compressed Image (JPEG, Quality: {quality}, File Size: {file_size:.2f} KB, PSNR: {psnr_value:.2f} dB)"
            explanation = """
            **JPEG Compression:**  A lossy compression method that reduces file size by discarding some image data.
            Higher quality settings result in larger file sizes but better image fidelity.

            **💡 Analogy:** Think of summarizing a long story. You keep the main points (important image data), 
            but leave out some details (less important data). 
            """
        elif compression_type == "PNG":
            compression_level = st.slider("PNG Compression Level (0-9)", 0, 9, 3)
            _, encoded_img = cv2.imencode(
                ".png", img_array, [cv2.IMWRITE_PNG_COMPRESSION, compression_level]
            )
            compressed_img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)  # Decode as color (RGB)

            file_size = len(encoded_img) / 1024
            psnr_value = psnr(img_array, compressed_img)
            caption = f"Compressed Image (PNG, Compression Level: {compression_level}, File Size: {file_size:.2f} KB, PSNR: {psnr_value:.2f} dB)"
            explanation = """
            **PNG Compression:** A lossless compression method that reduces file size without losing any image data.
            Higher compression levels achieve smaller file sizes.

            **💡 Analogy:** Imagine packing a suitcase very carefully. You can fit more clothes (image data) by organizing them efficiently, 
            without discarding any items.
            """

        col1, col2 = st.columns(2)
        with col1:
            st.image(img_array, caption="Original Image", use_column_width=True)
        with col2:
            st.image(compressed_img, caption=caption, use_column_width=True)

        st.markdown(explanation)

        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            st.button("Previous", on_click=prev_page)
        with col3:
            st.button("Next", on_click=next_page)
    else:
        st.write("Please upload an image on the first page.")
        st.button("Previous", on_click=prev_page)


def page_color_conversion():
    st.header("Color Space Conversion 🌈")
    if st.session_state.uploaded_image is not None:
        img_array = np.array(st.session_state.uploaded_image)
        conversion_type = st.selectbox("Select Conversion:", ["RGB to Grayscale", "RGB to HSV"])

        if conversion_type == "RGB to Grayscale":
            converted_img = utils.grayscale(img_array)
            explanation = """
            **RGB to Grayscale:** Converts a color image to grayscale by averaging the red, green, and blue channels. 

            **💡 Analogy:** Imagine looking at a colorful scene through a black-and-white filter. 
            You lose the color information but retain the brightness variations.

            **🧮 Formula (Luminosity Method):**
            One common method to calculate grayscale intensity (Y) from RGB values (R, G, B) is:

            $$
            Y = 0.299R + 0.587G + 0.114B 
            $$
            """
        elif conversion_type == "RGB to HSV":
            converted_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            explanation = """
            **RGB to HSV:**  Converts a color image from the RGB color space to the HSV (Hue, Saturation, Value) space.
            HSV represents colors in terms of their hue (color shade), saturation (color intensity), and value (brightness). 

            **💡 Analogy:**  Think of an artist's color wheel. Hue is the position on the wheel, saturation is how vivid the color is, 
            and value is how light or dark the color is.
            """

        col1, col2 = st.columns(2)
        with col1:
            st.image(img_array, caption="Original Image", use_column_width=True)
        with col2:
            st.image(
                converted_img,
                caption=f"Converted Image ({conversion_type})",
                use_column_width=True,
            )

        st.markdown(explanation)

        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            st.button("Previous", on_click=prev_page)
        with col3:
            st.button("Next", on_click=next_page)
    else:
        st.write("Please upload an image on the first page.")
        st.button("Previous", on_click=prev_page)


def page_histogram_equalization():
    st.header("Histogram Equalization 📊")
    st.markdown(
        """
    Histogram equalization is a technique used to enhance the contrast of an image by redistributing pixel intensities. 
    It works by:
    - **Analyzing the image's histogram:** A histogram shows the distribution of pixel values (how many pixels are at each intensity level).
    - **Remapping the pixel values:**  The pixel values are reassigned based on the cumulative distribution function of the histogram, resulting in a more uniform distribution of intensities. 

    **This often leads to:**
    - **Increased detail in dark or bright areas**
    - **A more balanced overall contrast in the image.**
    """
    )

    if st.session_state.uploaded_image is not None:
        img_array = np.array(st.session_state.uploaded_image)
        if len(img_array.shape) == 3:  # Check if image is color
            gray_img = utils.grayscale(img_array)
        else:
            gray_img = img_array
        equalized_img = utils.histogram_equalization(gray_img)

        col1, col2 = st.columns(2)
        with col1:
            st.image(gray_img, caption="Original Grayscale Image", use_column_width=True)
            fig, ax = plt.subplots()
            ax.hist(
                gray_img.flatten(), 256, [0, 256]
            )  # Specify bins for better visualization
            st.pyplot(fig)
            st.markdown(
                """
            **Original Histogram:** 
            This histogram shows that the majority of pixels are concentrated in the lower intensity levels, indicating low contrast.
            """
            )

        with col2:
            st.image(equalized_img, caption="Equalized Image", use_column_width=True)
            fig, ax = plt.subplots()
            ax.hist(equalized_img.flatten(), 256, [0, 256])  # Specify bins
            st.pyplot(fig)
            st.markdown(
                """
            **Equalized Histogram:**
            After equalization, the histogram shows a more even distribution of pixel values across the intensity range, indicating improved contrast.
            """
            )

        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            st.button("Previous", on_click=prev_page)
        with col3:
            st.button("Next", on_click=next_page)
    else:
        st.write("Please upload an image on the first page.")
        st.button("Previous", on_click=prev_page)


def page_segmentation():
    st.header("Image Segmentation ✂️")
    if st.session_state.uploaded_image is not None:
        img_array = np.array(st.session_state.uploaded_image)
        segmentation_type = st.selectbox(
            "Select Segmentation Method:", ["Otsu's Thresholding", "Simple Thresholding"]
        )

        if segmentation_type == "Otsu's Thresholding":
            if len(img_array.shape) == 3:
                gray_img = utils.grayscale(img_array)
            else:
                gray_img = img_array
            ret, segmented_img = cv2.threshold(
                gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            explanation = """
            **Otsu's Thresholding:**  Automatically determines the optimal threshold value to separate the image into foreground and background. 
            It minimizes the intra-class variance (variance within the foreground and background pixels).

            **💡 Analogy:** Imagine dividing a group of people into two teams based on their height, finding the height that best separates the tall group from the short group.
            """

        elif segmentation_type == "Simple Thresholding":
            threshold_value = st.slider("Threshold Value (0-255)", 0, 255, 127)
            if len(img_array.shape) == 3:
                gray_img = utils.grayscale(img_array)
            else:
                gray_img = img_array
            ret, segmented_img = cv2.threshold(
                gray_img, threshold_value, 255, cv2.THRESH_BINARY
            )
            explanation = """
            **Simple Thresholding:** Divides the image into two regions based on a user-defined threshold value. 
            Pixels above the threshold are assigned one value, and pixels below are assigned another.

            **💡 Analogy:** Imagine separating black and white marbles by rolling them down a ramp with a divider. 
            Marbles above the divider go one way, and marbles below go the other way. 
            """

        col1, col2 = st.columns(2)
        with col1:
            st.image(img_array, caption="Original Image", use_column_width=True)
        with col2:
            st.image(
                segmented_img,
                caption=f"Segmented Image ({segmentation_type})",
                use_column_width=True,
            )

        st.markdown(explanation)

        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            st.button("Previous", on_click=prev_page)
        with col3:
            st.button("Next", on_click=next_page)
    else:
        st.write("Please upload an image on the first page.")
        st.button("Previous", on_click=prev_page)


def page_fourier_transform():
    st.header("Fourier Transform 🌊")
    if st.session_state.uploaded_image is not None:
        img_array = np.array(st.session_state.uploaded_image)
        gray_img = utils.grayscale(img_array)
        magnitude_spectrum = utils.fourier_transform(gray_img)

        # Normalize the magnitude spectrum to 0-1 range for display
        magnitude_spectrum = (magnitude_spectrum - np.min(magnitude_spectrum)) / (
            np.max(magnitude_spectrum) - np.min(magnitude_spectrum)
        )

        col1, col2 = st.columns(2)
        with col1:
            st.image(gray_img, caption="Grayscale Image", use_column_width=True)
        with col2:
            st.image(
                magnitude_spectrum, caption="Magnitude Spectrum", use_column_width=True
            )

        st.markdown("""
        **Fourier Transform:**  Decomposes an image into its frequency components. 
        The magnitude spectrum visualizes the strength of these frequencies.

        **💡 Analogy:** Think of a music equalizer. The Fourier Transform is like separating a song into its different frequencies (bass, treble, etc.), 
        and the magnitude spectrum shows the amplitude of each frequency. 

        **🧮 Mathematical Representation:** The Fourier Transform of an image $f(x,y)$ is given by:

        $$
        F(u,v) = \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} f(x,y)e^{-2\pi i (ux + vy)} dx dy
        $$

        where:
        - $F(u,v)$ is the Fourier Transform at frequency $(u,v)$.
        - $f(x,y)$ is the image intensity at spatial coordinates $(x,y)$.
        """)

        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            st.button("Previous", on_click=prev_page)
        
    else:
        st.write("Please upload an image on the first page.")
        st.button("Previous", on_click=prev_page)

# --- Display the Current Page ---
display_page(st.session_state.page)
