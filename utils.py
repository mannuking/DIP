# utils.py

import cv2
import numpy as np
import random
from scipy import fftpack
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from sklearn.cluster import KMeans

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def sharpen(img, intensity):
    kernel = np.array([[-1, -1, -1], [-1, 9 + intensity - 1, -1], [-1, -1, -1]])
    return cv2.filter2D(img, -1, kernel)

def sobel_edge_detection(img):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    abs_grad_x = cv2.convertScaleAbs(sobelx)
    abs_grad_y = cv2.convertScaleAbs(sobely)
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return grad

def canny_edge_detection(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def median_filter(img, kernel_size):
    return cv2.medianBlur(img, kernel_size)

def fourier_transform(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    return magnitude_spectrum

def jpeg_compression(img, quality):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', img, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg

def add_gaussian_noise(img, mean, var):
    gaussian_noise = np.random.normal(mean, var**0.5, img.shape) * 255
    noisy_img = img + gaussian_noise.astype(np.int32)
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img

def add_salt_and_pepper_noise(img, prob):
    output = np.copy(img)
    thres = 1 - prob
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
    return output

def mean_filter(img, kernel_size):
    return cv2.blur(img, (kernel_size, kernel_size))

def histogram_equalization(img):
    return cv2.equalizeHist(img)

def reduce_bit_depth(img, bits):
    return (img >> (8 - bits)) << (8 - bits)

def apply_filter_realtime(frame, filter_type, **kwargs):
    if filter_type == "Gaussian Blur":
        return gaussian_blur(frame, kwargs.get('kernel_size', 5))
    elif filter_type == "Canny Edge Detection":
        return canny_edge_detection(grayscale(frame), kwargs.get('low_threshold', 50), kwargs.get('high_threshold', 150))
    else:
        return frame

def process_video(video_file, filter_type, **kwargs):
    cap = cv2.VideoCapture(video_file)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = apply_filter_realtime(frame, filter_type, **kwargs)
        frames.append(processed_frame)
    cap.release()
    return frames
