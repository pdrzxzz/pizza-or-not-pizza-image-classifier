# Features to be extracted from images
FEATURES = ['gray', 'color', 'lbp', 'sobel', 'hog']
# Size to which image will be resized (width, height)
IMAGE_SIZE = (128, 128)
# Parameters for Local Binary Patterns (LBP)
LBP_RADIUS = 1                  # Radius of circle for LBP
LBP_N_POINTS = 8 * LBP_RADIUS  # Number of points to consider in LBP
# Random state for reproducibility in splitting and models
RANDOM_STATE = 0