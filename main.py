import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Function to perform Gaussian Smoothing
def GaussianSmoothing(image, sigma):
    kernel_size = 5
    kernel = np.zeros((kernel_size, kernel_size))
    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel[i, j] = np.exp(-(i**2 + j**2) / (2 * sigma**2))
    kernel = kernel / np.sum(kernel)

    padding = kernel_size // 2
    padded_image = np.zeros(
        (image.shape[0] + 2 * padding, image.shape[1] + 2 * padding, image.shape[2]), dtype=image.dtype)
    padded_image[padding:-padding, padding:-padding, :] = image

    output = np.zeros_like(image)
    for c in range(image.shape[2]):
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                output[i, j, c] = np.sum(
                    padded_image[i:i + kernel_size, j:j + kernel_size, c] * kernel)

    return image, output

# Function to perform Sobel Edge Detection
def SobelEdgeDetection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kernelY = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    kernelX = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])

    padding = kernelX.shape[0] // 2
    padded_image = np.zeros(
        (gray.shape[0] + 2 * padding, gray.shape[1] + 2 * padding), dtype=np.uint8)
    padded_image[padding:-padding, padding:-padding] = gray

    output = np.zeros_like(gray)
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            summX = np.sum(
                padded_image[i:i + kernelX.shape[0], j:j + kernelX.shape[0]] * kernelX)
            summY = np.sum(
                padded_image[i:i + kernelX.shape[0], j:j + kernelX.shape[0]] * kernelY)
            output[i, j] = (summX ** 2 + summY ** 2) ** (1 / 2)

    return output

# Function to perform Laplacian Edge Detection
def LaplacianEdgeDetection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]])

    padding = kernel.shape[0] // 2
    padded_image = np.zeros(
        (gray.shape[0] + 2 * padding, gray.shape[1] + 2 * padding), dtype=np.uint8)
    padded_image[padding:-padding, padding:-padding] = gray

    output = np.zeros_like(gray)
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            summ = np.sum(
                padded_image[i:i + kernel.shape[0], j:j + kernel.shape[0]] * kernel)
            output[i, j] = summ * summ

    return output

# Main Streamlit App
def main():
    st.title("Explore Image Processing Effects")
    st.subheader("Author : Luka Gorgadze")
    st.write("Welcome! This web app allows you to apply various image processing effects on your uploaded images. "
             "Let's dive into the world of image magic!")

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Load the uploaded image
        image = Image.open(uploaded_image)
        image = np.array(image)

        st.header("Original Image")
        st.image(image, channels="RGB", use_column_width=True)

        st.subheader("Gaussian Smoothing")
        
        st.write("Gaussian Smoothing is an image filtering technique used to reduce noise and blur the image slightly. "
                 "It convolves the image with a Gaussian kernel to create a softer appearance.")

        # Choose the sigma value for Gaussian Smoothing
        sigma = st.slider("Select the Sigma value for Gaussian Smoothing", 1, 20, 5)
        colored, smoothed = GaussianSmoothing(image.copy(), sigma)
        st.image(smoothed, channels="RGB", use_column_width=True)

        st.subheader("Edge Detection")
        st.write("Edge Detection highlights the boundaries of objects in an image. "
                 "You can choose between Sobel Edge Detection and Laplacian Edge Detection.")

        # Choose the edge detection algorithm
        selected_algorithm = st.radio("Select an Edge Detection Algorithm", ("Sobel Edge Detection", "Laplacian Edge Detection"))

        if selected_algorithm == "Sobel Edge Detection":
            st.subheader("Sobel Edge Detection")
            st.write("Sobel Edge Detection uses two convolutional filters to detect the horizontal and vertical edges "
                     "in an image. The magnitude of the gradients indicates the strength of the edges.")
            edged = SobelEdgeDetection(smoothed)
            st.image(edged, channels="L", use_column_width=True)

        elif selected_algorithm == "Laplacian Edge Detection":
            st.subheader("Laplacian Edge Detection")
            st.write("Laplacian Edge Detection calculates the second derivative of the image to find regions of rapid "
                     "intensity changes. The resulting image highlights the edges and removes the details.")
            edged = LaplacianEdgeDetection(smoothed)
            st.image(edged, channels="L", use_column_width=True)

if __name__ == "__main__":
    main()
