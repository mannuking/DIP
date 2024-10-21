# Digital Image Processing Interactive Demo

This is an interactive web application built using Streamlit that demonstrates various Digital Image Processing (DIP) techniques. It allows users to upload an image and experiment with different DIP applications in a step-by-step, interactive manner. 

## Installation and Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/your-repo-name.git 
   ```
   (Replace `your-username` and `your-repo-name` with your actual GitHub username and repository name)

2. **Navigate to the Project Directory:**
   ```bash
   cd your-repo-name
   ```

3. **Create a Virtual Environment (Recommended):**
   ```bash
   python -m venv env
   ```

4. **Activate the Virtual Environment:**
   - On Windows:
     ```bash
     env\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source env/bin/activate
     ```

5. **Install the Required Packages:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the App

1. Make sure your virtual environment is activated.
2. In the terminal, navigate to the project directory.
3. Run the following command:
   ```bash
   streamlit run app.py 
   ```

4. The app will open in your web browser.

## DIP Applications 

### 1. Upload Image

- **Definition:**  Allows users to upload an image in JPG, PNG, or JPEG format to be used for the DIP experiments.

### 2. Bit Depth Adjustment

- **Definition:** Bit depth refers to the number of bits used to represent the color information of each pixel in an image. Adjusting the bit depth changes the number of colors or shades of gray that can be represented.

- **Analogy:** Imagine a paint set. A set with only a few colors (low bit depth) limits your options, while a set with many colors (high bit depth) offers more possibilities for subtle shading and detail.

- **Formula:** The number of colors (or gray levels) that can be represented by a given bit depth is:

   ```
   Number of Colors = 2^(Bit Depth)
   ```

- **Numerical Example:** An 8-bit image can represent 2^8 = 256 colors, while a 4-bit image can only represent 2^4 = 16 colors.

### 3. Image Filtering

- **Definition:** Image filtering is a technique used to modify the pixels of an image based on their neighboring pixels. It's used for tasks like blurring, sharpening, and edge detection.

- **Types of Filters:**
   - **Gaussian Blur:** Smooths the image by averaging pixel values within a kernel, reducing noise and detail. 
     - **Analogy:** Imagine rubbing a blurry piece of glass over the image. The larger the glass, the more blurred the image becomes.
   - **Sharpen:** Enhances edges and details in the image by increasing the contrast between neighboring pixels. 
     - **Analogy:** Think of tracing the outlines of objects in the image with a darker pen, making them stand out more.
   - **Sobel Edge Detection:** Detects edges by calculating gradients in horizontal and vertical directions.
     - **Analogy:** Imagine an artist sketching the image, focusing only on the outlines and boundaries of objects.
   - **Canny Edge Detection:** A multi-step edge detection algorithm that identifies edges more accurately by suppressing noise. 
     - **Analogy:** Think of a sophisticated image editing software automatically selecting the most prominent edges in the image, like outlining important features for a graphic design.

### 4. Noise Addition and Removal

- **Definition:** Noise is random variation in pixel values, often unwanted. This section allows you to add noise and then apply filters to remove it. 

- **Types of Noise:**
    - **Gaussian Noise:** Adds random noise following a Gaussian (normal) distribution. 
    - **Salt & Pepper Noise:** Replaces random pixels with either black (pepper) or white (salt) values.

- **Types of Filters for Noise Removal:**
    - **Mean Filter:** Averages pixel values within a kernel, smoothing out noise.
        - **Analogy:** Think of blending colors on a canvas. A mean filter mixes the colors of neighboring pixels.
    - **Median Filter:** Replaces the center pixel with the median value within the kernel.  Effective at removing impulsive noise (salt & pepper).
        - **Analogy:** Imagine picking the "middle" color from a group of neighboring pixels to represent the center pixel.

- **Analogy (Overall Process):** Imagine a photo that has been damaged (noise). Restoration filters are like tools used to repair the photo. Some tools might blend colors (mean filter) to smooth out scratches, while others might replace damaged spots with colors from nearby areas (median filter). 

### 5. Image Compression and Quality

- **Definition:** Image compression reduces the file size of an image.  Lossy compression discards some data, while lossless compression preserves all data.

- **Types of Compression:**
    - **JPEG:** Lossy compression, often used for photographs. 
        - **Analogy:** Think of summarizing a long story. You keep the main points (important image data), but leave out some details (less important data). 
    - **PNG:** Lossless compression, often used for images with sharp edges and text.
        - **Analogy:** Imagine packing a suitcase very carefully. You can fit more clothes (image data) by organizing them efficiently, without discarding any items.

- **Numerical Example:** A JPEG image can be compressed to a smaller file size than a PNG image for the same quality, but some details will be lost.

### 6. Color Space Conversion

- **Definition:** Color space conversion changes how color information is represented (e.g., RGB, HSV).

- **Types of Conversions:**
    - **RGB to Grayscale:** Converts a color image to grayscale.
        - **Analogy:** Imagine looking at a colorful scene through a black-and-white filter. You lose the color information but retain the brightness variations.
        - **Formula (Luminosity Method):**  A common formula for converting RGB to grayscale is: 
        ```
        Y = 0.299R + 0.587G + 0.114B
        ```
        where Y is the grayscale intensity, and R, G, B are the red, green, and blue channel values.
    - **RGB to HSV:** Converts an image to the Hue, Saturation, Value color space.
        - **Analogy:** Think of an artist's color wheel. Hue is the position on the wheel, saturation is how vivid the color is, and value is how light or dark the color is.

### 7. Histogram Equalization

- **Definition:** Enhances image contrast by redistributing pixel intensities to make the histogram more uniform.

- **Analogy:** Imagine a room with dim lighting. Histogram equalization is like turning up the lights so that all objects are more clearly visible.

- **Example:** An image with low contrast will have a histogram where most pixel values are concentrated in a narrow range. After equalization, the histogram will be spread out more evenly, resulting in improved contrast.

### 8. Image Segmentation

- **Definition:** Divides an image into multiple segments (regions) based on certain criteria.

- **Methods:**
    - **Otsu's Thresholding:** Automatically finds the best threshold value to separate the image into foreground and background. 
        - **Analogy:** Imagine dividing a group of people into two teams based on their height, finding the height that best separates the tall group from the short group.
    - **Simple Thresholding:**  Divides the image based on a user-defined threshold.
        - **Analogy:** Imagine separating black and white marbles by rolling them down a ramp with a divider. Marbles above the divider go one way, and marbles below go the other way. 

### 9. Fourier Transform

- **Definition:** Decomposes an image into its frequency components, showing how much of each frequency is present.

- **Analogy:** Think of a music equalizer. The Fourier Transform is like separating a song into its different frequencies (bass, treble, etc.), and the magnitude spectrum shows the amplitude of each frequency. 

- **Mathematical Representation:** 
    ```
    F(u,v) = ∫(-∞ to ∞) ∫(-∞ to ∞) f(x,y)e^(-2πi(ux + vy)) dx dy
    ```
    where:
    -  $F(u,v)$ is the Fourier Transform at frequency (u,v).
    - $f(x,y)$ is the image intensity at spatial coordinates (x,y). 

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License.


