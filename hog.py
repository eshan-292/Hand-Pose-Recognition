import numpy as np
import cv2

def compute_gradients(image):
    # Compute gradients using Sobel operator
    dx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=1)
    dy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=1)
    
    # Compute magnitude and orientation of gradients
    magnitude, orientation = cv2.cartToPolar(dx, dy, angleInDegrees=True)
    
    return magnitude, orientation

def compute_histogram(cell_magnitude, cell_orientation, num_bins=9):
    # Create histogram bins
    bins = np.linspace(0, 180, num_bins+1)
    
    # Initialize histogram
    histogram = np.zeros(num_bins)
    
    # Calculate histogram
    for i in range(num_bins):
        histogram[i] = np.sum(cell_magnitude[np.where((cell_orientation >= bins[i]) & (cell_orientation < bins[i+1]))])
    
    return histogram

def block_normalization(histograms):
    # Flatten the histogram matrix
    flat_histograms = histograms.flatten()
    
    # Calculate the block normalization factor
    normalization_factor = np.sqrt(np.sum(flat_histograms**2) + 1e-5)
    
    # Normalize blocks
    normalized_blocks = flat_histograms / normalization_factor
    
    return normalized_blocks

def extract_hog_features(image, win_size=(64, 64), block_size=(16, 16), block_stride=(8, 8), cell_size=(8, 8), num_bins=9):
    win_size = (64, 64)  # Window size for HOG descriptor
    block_size = (16, 16)  # Block size for HOG descriptor
    block_stride = (8, 8)  # Block stride for HOG descriptor
    cell_size = (8, 8)  # Cell size for HOG descriptor
    num_bins = 9  # Number of bins for HOG descriptor
    
    # Compute gradients
    gradient_magnitude, gradient_orientation = compute_gradients(image)
    
    # Calculate the number of cells per block
    cells_per_block = (block_size[0] // cell_size[0], block_size[1] // cell_size[1])
    
    # Calculate the number of cells in x and y directions
    num_cells_x = (win_size[1] - block_size[1]) // block_stride[1] + 1
    num_cells_y = (win_size[0] - block_size[0]) // block_stride[0] + 1
    
    # Initialize HOG feature vector
    hog_features = []
    
    # Iterate over blocks in the image
    for y in range(num_cells_y):
        for x in range(num_cells_x):
            # Calculate the starting point of the block
            start_x = x * block_stride[1]
            start_y = y * block_stride[0]
            
            # Compute gradients and orientations for the block
            block_magnitude = gradient_magnitude[start_y : start_y + block_size[0], 
                                                  start_x : start_x + block_size[1]]
            block_orientation = gradient_orientation[start_y : start_y + block_size[0], 
                                                      start_x : start_x + block_size[1]]
            
            # Initialize histogram for the block
            block_histogram = np.zeros((cells_per_block[0], cells_per_block[1], num_bins))
            
            # Iterate over cells in the block
            for i in range(cells_per_block[0]):
                for j in range(cells_per_block[1]):
                    # Compute histogram for the cell
                    cell_magnitude = block_magnitude[i * cell_size[0] : (i + 1) * cell_size[0], 
                                                     j * cell_size[1] : (j + 1) * cell_size[1]]
                    cell_orientation = block_orientation[i * cell_size[0] : (i + 1) * cell_size[0], 
                                                         j * cell_size[1] : (j + 1) * cell_size[1]]
                    cell_histogram = compute_histogram(cell_magnitude, cell_orientation, num_bins)
                    
                    # Assign histogram to the block
                    block_histogram[i, j, :] = cell_histogram
            
            # Normalize block histogram
            block_histogram_normalized = block_normalization(block_histogram)
            
            # Flatten block histogram and append to HOG features
            hog_features.append(block_histogram_normalized.flatten())
    
    # Convert HOG features to numpy array
    hog_features = np.array(hog_features)

    # reshape the HOG features to match the OpenCV HOG features
    hog_features = hog_features.reshape(-1, 1)
    
    return hog_features



# def extract_hog_features_opencv(image):
#     win_size = (64, 64)  # Window size for HOG descriptor
#     block_size = (16, 16)  # Block size for HOG descriptor
#     block_stride = (8, 8)  # Block stride for HOG descriptor
#     cell_size = (8, 8)  # Cell size for HOG descriptor
#     num_bins = 9  # Number of bins for HOG descriptor
    
#     # Create HOG descriptor object
#     hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)
    
#     # Compute HOG features
#     hog_features = hog.compute(image)
    
#     return hog_features


# # Example usage:
# image = np.random.randint(0, 255, (64, 64)).astype(np.uint8)  # Random grayscale image
# hog_features_custom = extract_hog_features(image)
# hog_features_opencv = extract_hog_features_opencv(image)

# # hog_features_opencv = hog_features_opencv.squeeze().reshape(hog_features_custom.shape)
# # convert the custom HOG features to the same shape as OpenCV HOG features
# # hog_features_custom = hog_features_custom.reshape(hog_features_opencv.shape)



# print("Custom HOG Features shape:", hog_features_custom.shape)
# print("OpenCV HOG Features shape:", hog_features_opencv.shape)

# print("Custom HOG Features:", hog_features_custom)
# print("OpenCV HOG Features:", hog_features_opencv)

# # Compare the results
# if np.allclose(hog_features_custom, hog_features_opencv):
#     print("Custom HOG features match OpenCV HOG features.")
# else:
#     print("Custom HOG features do not match OpenCV HOG features.")


