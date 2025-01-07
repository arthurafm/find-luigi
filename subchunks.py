import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Load images
imgs_rgb = []
imgs_greyscale = []
for file in [f'test{i}.png' for i in range(6)]:
    img = cv.imread(file, cv.COLOR_RGB2BGR)
    imgs_rgb.append(img)
    imgs_greyscale.append(cv.cvtColor(img, cv.COLOR_BGR2GRAY))

imgs2 = imgs_greyscale.copy()

# Load the template
template_original = cv.imread('icon.png', cv.IMREAD_GRAYSCALE)

# Define minimum confidence threshold
CONFIDENCE_THRESHOLD = 0.89

# Template matching methods
methods = ['TM_CCOEFF_NORMED', 'TM_CCORR_NORMED', 'TM_SQDIFF_NORMED']

def divide_template(template, divisions=2):
    """Divide the template into smaller chunks."""
    w, h = template.shape
    chunk_h = h // divisions
    chunk_w = w // divisions
    chunks = []
    for i in range(divisions):
        for j in range(divisions):
            chunk = template[i * chunk_h:(i + 1) * chunk_h, j * chunk_w:(j + 1) * chunk_w]
            chunks.append((chunk, (j * chunk_w, i * chunk_h)))  # Store chunk with its offset
    return chunks

# Divide the template into chunks before resizing
chunks = divide_template(template_original, divisions=2)

# Resize the template
template = cv.resize(template_original, (70, 89))
w, h = template.shape[::-1]

for i in range(6):
    print(f'Image nº{i+1}')
    for meth in methods:
        img = imgs2[i].copy()
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

        print(f'Full template confidence: {confidence}')
        
        if confidence < CONFIDENCE_THRESHOLD:
            print('Confidence below threshold. Using pre-divided template chunks...')
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
            
            print(f'Best chunk confidence: {confidence}')
        else:
            bottom_right = (top_left[0] + w, top_left[1] + h)

        # Draw rectangle around the detected region
        img_rgb = imgs_rgb[i].copy()
        cv.rectangle(img_rgb, top_left, bottom_right, (0, 0, 255), 2)

        # Show results
        plt.imshow(cv.cvtColor(img_rgb, cv.COLOR_BGR2RGB))
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        plt.suptitle(meth)
        plt.show()
