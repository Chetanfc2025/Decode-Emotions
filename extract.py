import numpy as np
import cv2
import math

# =============== Configuration Constants ===============
ANCHOR_POINT = 6000
MIDZONE_THRESHOLD = 15000
MIN_HANDWRITING_HEIGHT_PIXEL = 20

# Features are defined here as global variables
BASELINE_ANGLE = 0.0
TOP_MARGIN = 0.0
LETTER_SIZE = 0.0
LINE_SPACING = 0.0
WORD_SPACING = 0.0

# =============== Image Processing Functions ===============
def bilateral_filter(image, diameter=5):
    """Apply bilateral filtering while preserving edges."""
    return cv2.bilateralFilter(image, diameter, 50, 50)

def median_filter(image, kernel_size=5):
    """Apply median filtering for noise reduction."""
    return cv2.medianBlur(image, kernel_size)

def threshold_image(image, threshold_value=120):
    """Convert to grayscale and apply inverted binary threshold."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
    return thresh

def dilate_image(image, kernel_size=(5, 100)):
    """Dilate objects in the image using a rectangular kernel."""
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.dilate(image, kernel, iterations=1)

def erode_image(image, kernel_size=(5, 100)):
    """Erode objects in the image using a rectangular kernel."""
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.erode(image, kernel, iterations=1)

# =============== Contour Straightening Function ===============
def straighten_contours(image):
    """Straighten contours horizontally for better projection analysis."""
    global BASELINE_ANGLE

    filtered = bilateral_filter(image, diameter=3)
    thresh = threshold_image(filtered, threshold_value=120)
    dilated = dilate_image(thresh)

    # Updated for OpenCV >= 4.x compatibility
    contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    angle_sum = 0.0
    contour_count = 0

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Skip contours that are too small or not lines
        if h > w or h < MIN_HANDWRITING_HEIGHT_PIXEL:
            continue

        roi = image[y:y+h, x:x+w]

        if w < image.shape[1] / 2:
            roi[:] = 255
            image[y:y+h, x:x+w] = roi
            continue

        rect = cv2.minAreaRect(contour)
        angle = rect[2]
        if angle < -45.0:
            angle += 90.0

        center = ((x + w) // 2, (y + h) // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1)
        rotated_roi = cv2.warpAffine(roi, rotation_matrix, (w, h), borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(255, 255, 255))

        image[y:y+h, x:x+w] = rotated_roi
        angle_sum += angle
        contour_count += 1

    if contour_count > 0:
        BASELINE_ANGLE = angle_sum / contour_count

    return image

# =============== Projection Functions ===============
def horizontal_projection(image):
    """Calculate horizontal projection profile."""
    return np.sum(image == 255, axis=1).tolist()

def vertical_projection(image):
    """Calculate vertical projection profile."""
    return np.sum(image == 255, axis=0).tolist()

# =============== Line Extraction Function ===============
def extract_lines(image):
    """Extract lines using horizontal projection analysis."""
    global LETTER_SIZE
    global LINE_SPACING
    global TOP_MARGIN

    filtered_image = bilateral_filter(image)
    thresh_image = threshold_image(filtered_image)

    hp_list = horizontal_projection(thresh_image)

    top_margin_count = sum(1 for value in hp_list if value <= 255)

    lines = []
    line_start_index = None

    for i in range(len(hp_list)):
        if hp_list[i] > 255:  # Start of a line
            if line_start_index is None:
                line_start_index = i
        elif hp_list[i] <= 255:  # End of a line
            if line_start_index is not None:
                lines.append((line_start_index, i))
                line_start_index = None

    # Calculate features (LETTER_SIZE and LINE_SPACING)
    if lines:
        line_heights = [end - start for start, end in lines]
        LETTER_SIZE = np.mean(line_heights)
        LINE_SPACING_RATIO = np.mean([lines[i+1][0] - lines[i][1] for i in range(len(lines)-1)])
        TOP_MARGIN_RATIO = top_margin_count / LETTER_SIZE if LETTER_SIZE > 0 else 0

        LINE_SPACING_RATIO /= LETTER_SIZE if LETTER_SIZE > 0 else LINE_SPACING_RATIO

        TOP_MARGIN_RATIO /= LETTER_SIZE if LETTER_SIZE > 0 else TOP_MARGIN_RATIO

        LINE_SPACING_RATIO
