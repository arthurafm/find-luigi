import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Load the template
template_original = cv.imread('luigi.png', cv.IMREAD_GRAYSCALE)

mario_template = cv.imread('mario.png', cv.IMREAD_GRAYSCALE)
wario_template = cv.imread('wario.png', cv.IMREAD_GRAYSCALE)
yoshi_template = cv.imread('yoshi.png', cv.IMREAD_GRAYSCALE)

# Define minimum confidence threshold
CONFIDENCE_THRESHOLD = 0.98

# Template matching methods
# methods = ['TM_CCOEFF_NORMED', 'TM_CCORR_NORMED', 'TM_SQDIFF_NORMED']
methods = [ 'TM_CCORR_NORMED' ]

def divide_template(template, divisions=2):
    """Divide the template into smaller chunks."""
    w, h = template.shape
    chunk_h = h // divisions
    chunk_w = w // divisions
    chunks = []
    for i in range(divisions):
        for j in range(divisions):
            chunk = template[i * chunk_h:(i + 1) * chunk_h, j * chunk_w:(j + 1) * chunk_w]
            if np.count_nonzero(chunk) >= (chunk.size / 2):
                chunks.append((chunk, (j * chunk_w, i * chunk_h)))  # Store chunk with its offset
    return chunks

# Divide the template into chunks before resizing
chunks = divide_template(template_original, divisions=2)

# Filter empty chunks
chunks = [ chunk for chunk in chunks if chunk[0].size != 0 ]

# Filter chunks too similar to chunks of other icons
method = getattr(cv, methods[0])
new_chunks = set()
other_templates = [ mario_template, wario_template, yoshi_template ]

for other_template in other_templates:
    for chunk, offset in chunks:
        res = cv.matchTemplate(other_template, chunk, method)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

        if method == cv.TM_SQDIFF_NORMED:
            top_left = min_loc
            confidence = 1 - min_val
        else:
            top_left = max_loc
            confidence = max_val

        if confidence < 0.89:
            new_chunks.add((chunk, offset))

chunks = list(new_chunks)

# Resize the template
template = cv.resize(template_original, (70, 89))
w, h = template.shape[::-1]

# Input and output video paths
input_video_path = 'easy0.mp4'
output_video_path = 'easy0-99_2.avi'

# Open the input video
cap = cv.VideoCapture(input_video_path)
if not cap.isOpened():
    print("Error: Unable to open video file.")
    exit()

# Get video properties
frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv.CAP_PROP_FPS))

# Define the codec and create VideoWriter
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    for meth in methods:
        img = frame_gray.copy()
        method = getattr(cv, meth)

        # Apply template matching for the full template
        res = cv.matchTemplate(img, template, method)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

        if method == cv.TM_SQDIFF_NORMED:
            top_left = min_loc
            confidence = 1 - min_val
        else:
            top_left = max_loc
            confidence = max_val

        if confidence < CONFIDENCE_THRESHOLD:
            # Perform matching for each pre-divided chunk
            for chunk, offset in chunks:
                chunk_resized = cv.resize(chunk, (chunk.shape[1] * w // template_original.shape[1], 
                                                    chunk.shape[0] * h // template_original.shape[0]))
                res_chunk = cv.matchTemplate(img, chunk_resized, method)
                min_val_chunk, max_val_chunk, min_loc_chunk, max_loc_chunk = cv.minMaxLoc(res_chunk)

                if method == cv.TM_SQDIFF_NORMED:
                    chunk_top_left = min_loc_chunk
                    chunk_confidence = 1 - min_val_chunk
                else:
                    chunk_top_left = max_loc_chunk
                    chunk_confidence = max_val_chunk

                if chunk_confidence > confidence:
                    confidence = chunk_confidence
                    top_left = (chunk_top_left[0] + offset[0], chunk_top_left[1] + offset[1])
                    bottom_right = (top_left[0] + chunk_resized.shape[1], top_left[1] + chunk_resized.shape[0])
        else:
            bottom_right = (top_left[0] + w, top_left[1] + h)

    # Draw rectangle around the detected region
    cv.rectangle(frame, top_left, bottom_right, (255, 0, 255), 2)

    # Write the frame to the output video
    out.write(frame)

# Release resources
cap.release()
out.release()
print(f"Processed video saved as {output_video_path}")
