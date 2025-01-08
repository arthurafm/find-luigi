import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import sys

# Load the template
template_original = cv.imread('assets/luigi.png', cv.IMREAD_GRAYSCALE)

mario_template = cv.imread('assets/mario.png', cv.IMREAD_GRAYSCALE)
wario_template = cv.imread('assets/wario.png', cv.IMREAD_GRAYSCALE)
yoshi_template = cv.imread('assets/yoshi.png', cv.IMREAD_GRAYSCALE)

# Define minimum confidence threshold
FILTER_CHUNKS_CONFIDENCE_THRESHOLD = 0.98
CONFIDENCE_WITHOUT_CHUNKS_THRESHOLD = 0.90
CONFIDENCE_CHUNKS_2_THRESHOULD = 0.96
CONFIDENCE_CHUNKS_4_THRESHOULD = 0.98
CONFIDENCE_CHUNKS_8_THRESHOULD = 0.985

# Template matching method
method_name = 'TM_CCORR_NORMED'
method = getattr(cv, method_name)

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

    # Filter empty chunks
    chunks = [ chunk for chunk in chunks if chunk[0].size != 0 ]

    # Filter chunks too similar to chunks of other icons
    new_chunks = set()

    for other_template in [ mario_template, wario_template, yoshi_template ]:
        for i in range(len(chunks)):
            chunk, offset = chunks[i]

            res = cv.matchTemplate(other_template, chunk, method)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

            confidence = max_val

            if confidence > FILTER_CHUNKS_CONFIDENCE_THRESHOLD:
                new_chunks.add(i)

    filtered_chunks = []
    for i, value in enumerate(chunks):
        if i not in new_chunks:
            filtered_chunks.append(value)

    return filtered_chunks

# Divide the template into chunks before resizing
chunks_2 = divide_template(template_original, divisions=2)
chunks_4 = divide_template(template_original, divisions=4)
chunks_8 = divide_template(template_original, divisions=8)

# Resize the template
template = cv.resize(template_original, (70, 89))
w, h = template.shape[::-1]

frame = cv.imread(f'input_images/{sys.argv[1]}.png', cv.COLOR_RGB2BGR)

frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

img = frame_gray.copy()

# Apply template matching for the full template
res = cv.matchTemplate(img, template, method)
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
confidence = max_val

bb_votes = []

if confidence < CONFIDENCE_WITHOUT_CHUNKS_THRESHOLD:
    bb_votes.append((top_left, bottom_right))

# Perform matching for each pre-divided chunk
for chunks in [ chunks_2, chunks_4, chunks_8 ]:
    for chunk, offset in chunks:
        chunk_resized = cv.resize(chunk, (chunk.shape[1] * w // template_original.shape[1], 
                                            chunk.shape[0] * h // template_original.shape[0]))
        res_chunk = cv.matchTemplate(img, chunk_resized, method)
        min_val_chunk, max_val_chunk, min_loc_chunk, max_loc_chunk = cv.minMaxLoc(res_chunk)

        chunk_top_left = max_loc_chunk
        chunk_confidence = max_val_chunk

        doChunkVote = False

        if len(chunks) == 3:
            if chunk_confidence > CONFIDENCE_CHUNKS_2_THRESHOULD:
                doChunkVote = True
        elif len(chunks) == 8:
            if chunk_confidence > CONFIDENCE_CHUNKS_4_THRESHOULD:
                doChunkVote = True
        else:
            if chunk_confidence > CONFIDENCE_CHUNKS_8_THRESHOULD:
                doChunkVote = True

        if doChunkVote:
            confidence = chunk_confidence
            top_left = (chunk_top_left[0] - offset[0] * h // template_original.shape[1], chunk_top_left[1] - offset[1] * w // template_original.shape[0])
            bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])
            bb_votes.append((top_left, bottom_right))

overlap_counts = {}

for i, box1 in enumerate(bb_votes):
    count = 0
    for j, box2 in enumerate(bb_votes):
        if i != j:
            if (abs(box1[0][0] - box2[0][0]) <= 5
                or abs(box1[1][0] - box2[1][0]) <= 5
                or (box1[0][0] >= box2[0][0] and box1[0][0] <= box2[1][0])
                or (box2[0][0] >= box1[0][0] and box2[0][0] <= box1[1][0])):
                if (abs(box1[0][1] - box2[0][1]) <= 5
                    or abs(box1[1][1] - box2[1][1]) <= 5
                    or (box1[0][1] >= box2[0][1] and box1[0][1] <= box2[1][1])
                    or (box2[0][1] >= box1[0][1] and box2[0][1] <= box1[1][1])):
                    count += 1
    overlap_counts[i] = count

max_overlaps = max(overlap_counts.values())
winner_bbs = [bb_votes[idx] for idx, count in overlap_counts.items() if count == max_overlaps]

sum_top_left_0 = 0
sum_top_left_1 = 0
sum_bottom_right_0 = 0
sum_bottom_right_1 = 0
for winner_bb in winner_bbs:
    sum_top_left_0 += winner_bb[0][0]
    sum_top_left_1 += winner_bb[0][1]
    sum_bottom_right_0 += winner_bb[1][0]
    sum_bottom_right_1 += winner_bb[1][1]

top_left = (sum_top_left_0 // len(winner_bbs), sum_top_left_1 // len(winner_bbs))
bottom_right = (sum_bottom_right_0 // len(winner_bbs), sum_bottom_right_1 // len(winner_bbs))

# Draw rectangle around the detected region
cv.rectangle(frame, top_left, bottom_right, (255, 0, 255), 2)

# Debug frame-to-frame
plt.imshow(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
plt.show()
cv.imwrite(f'output_images/{sys.argv[1]}.png', frame)