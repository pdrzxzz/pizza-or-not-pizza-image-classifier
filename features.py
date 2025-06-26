import numpy as np
from io import BytesIO

from skimage import img_as_ubyte
from imageio.v3 import imread
from skimage.transform import resize
from skimage.feature import local_binary_pattern, hog
from skimage import filters
from skimage.color import rgb2gray
from skimage.color import rgba2rgb

# List of features to be extracted from the image
FEATURES = ['gray', 'color', 'lbp', 'sobel', 'hog']
# Target size for image resizing (width, height)
IMAGE_SIZE = (128, 128)
# Parameters for Local Binary Patterns (LBP)
LBP_RADIUS = 1
LBP_N_POINTS = 8 * LBP_RADIUS

def load_and_resize_image(uploaded_file, as_gray=True):
    """
    Load an image from a Streamlit UploadedFile object, convert to grayscale if requested,
    resize to a fixed size, and return as a NumPy array.

    Parameters:
    - uploaded_file (UploadedFile): File object from Streamlit file_uploader.
    - as_gray (bool): If True, convert the image to grayscale.

    Returns:
    - np.ndarray: Resized (and optionally grayscale) image.
    """
    # Read image bytes from the uploaded file and decode to numpy array
    image = imread(BytesIO(uploaded_file.read()))

    # If image has alpha channel, convert RGBA to RGB
    if image.ndim == 3 and image.shape[-1] == 4:
        image = rgba2rgb(image)

    # Convert to grayscale if requested and image has color channels
    if as_gray and image.ndim == 3:
        image = rgb2gray(image)

    # Resize image to fixed size with anti-aliasing
    image = resize(image, IMAGE_SIZE, anti_aliasing=True)

    return image


def extract_color_histograms(image):
    """
    Compute concatenated 256-bin histograms for each RGB channel of a color image.

    Parameters:
    - image (np.ndarray): Color image with pixel values normalized between 0 and 1.

    Returns:
    - np.ndarray: Concatenated histogram vector of length 768 (256 bins * 3 channels).
    """
    # If grayscale image is given, replicate channels to make it 3-channel
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)

    # Compute histogram per channel over the range [0,1]
    histograms = [
        np.histogram(image[:, :, channel], bins=256, range=(0, 1))[0]
        for channel in range(3)
    ]

    # Concatenate histograms of all channels into a single feature vector
    concatenated_hist = np.concatenate(histograms)
    return concatenated_hist


def extract_lbp_features(gray_image):
    """
    Calculate Local Binary Pattern (LBP) histogram features for a grayscale image.

    Parameters:
    - gray_image (np.ndarray): Grayscale image array.

    Returns:
    - np.ndarray: Normalized histogram of LBP patterns.
    """
    # Convert image to unsigned 8-bit format (required by LBP function)
    image_uint8 = img_as_ubyte(gray_image)

    # Compute LBP with 'uniform' method
    lbp = local_binary_pattern(image_uint8, LBP_N_POINTS, LBP_RADIUS, method="uniform")

    # Number of bins for histogram depends on LBP parameters
    n_bins = LBP_N_POINTS + 2

    # Calculate normalized histogram over the LBP patterns
    lbp_hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
    return lbp_hist


def extract_features_from_image(uploaded_file):
    """
    Extract selected features from an uploaded image file and return a concatenated feature vector.

    Parameters:
    - uploaded_file (UploadedFile): File object from Streamlit file_uploader.

    Returns:
    - np.ndarray: Concatenated feature vector of all selected features.
    """
    # Initialize dictionary to hold features arrays
    feature_arrays = {feature: [] for feature in FEATURES}

    # Load grayscale image for features that require it
    gray = load_and_resize_image(uploaded_file, as_gray=True)

    # Extract raw grayscale pixels if requested
    if "gray" in FEATURES:
        feature_arrays["gray"].append(gray.flatten())

    # Extract color histograms (requires color image)
    if "color" in FEATURES:
        # Reload the file, because read() exhausts the stream
        uploaded_file.seek(0)  # Reset stream position to start
        color_img = load_and_resize_image(uploaded_file, as_gray=False)
        feature_arrays["color"].append(extract_color_histograms(color_img))

    # Extract Local Binary Pattern histogram features
    if "lbp" in FEATURES:
        feature_arrays["lbp"].append(extract_lbp_features(gray))

    # Extract Sobel edge features
    if "sobel" in FEATURES:
        sobel_img = filters.sobel(gray)
        feature_arrays["sobel"].append(sobel_img.flatten())

    # Extract Prewitt edge features (optional, add 'prewitt' to FEATURES if used)
    if "prewitt" in FEATURES:
        prewitt_img = filters.prewitt(gray)
        feature_arrays["prewitt"].append(prewitt_img.flatten())

    # Extract Histogram of Oriented Gradients (HOG) features
    if "hog" in FEATURES:
        hog_features = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
        feature_arrays["hog"].append(hog_features)

    # Convert feature lists to arrays and concatenate all feature vectors horizontally
    arrays = [np.array(feature_arrays[f]) for f in FEATURES if feature_arrays[f]]

    if len(arrays) > 1:
        concatenated_features = np.concatenate(arrays, axis=1)
    else:
        concatenated_features = arrays[0]

    return concatenated_features
