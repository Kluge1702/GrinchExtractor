import cv2
import numpy as np

def calculate_dynamic_threshold(gray_diff):
    median = np.median(gray_diff)
    std_dev = np.std(gray_diff)
    return median + 2 * std_dev

def find_and_remove_largest_grinch(image_with_grinches, image_without_grinches, output_cutout_path,
                                   output_bez_tla_path, output_bbox_path):
    img_with = cv2.imread(image_with_grinches)
    img_without = cv2.imread(image_without_grinches)

    difference = np.abs(img_with.astype(np.int16) - img_without.astype(np.int16))
    gray_diff = np.mean(difference, axis=2).astype(np.uint8)

    threshold = calculate_dynamic_threshold(gray_diff)

    binary_mask = (gray_diff > threshold).astype(np.uint8) * 255

    height, width = binary_mask.shape
    cleaned_mask = np.zeros_like(binary_mask)

    for y in range(height):
        for x in range(width):
            if binary_mask[y, x] == 255:
                stack = [(y, x)]
                region = []

                while stack:
                    cy, cx = stack.pop()
                    if binary_mask[cy, cx] == 255:
                        binary_mask[cy, cx] = 0
                        region.append((cy, cx))

                        for ny, nx in [(cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)]:
                            if 0 <= ny < height and 0 <= nx < width and binary_mask[ny, nx] == 255:
                                stack.append((ny, nx))

                if len(region) > 50:
                    for ry, rx in region:
                        cleaned_mask[ry, rx] = 255

    labeled = np.zeros_like(cleaned_mask, dtype=np.int32)
    label = 1
    regions = []

    for y in range(height):
        for x in range(width):
            if cleaned_mask[y, x] == 255 and labeled[y, x] == 0:
                stack = [(y, x)]
                region = []

                while stack:
                    cy, cx = stack.pop()
                    if labeled[cy, cx] == 0 and cleaned_mask[cy, cx] == 255:
                        labeled[cy, cx] = label
                        region.append((cy, cx))

                        for ny, nx in [(cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)]:
                            if 0 <= ny < height and 0 <= nx < width and cleaned_mask[ny, nx] == 255 and labeled[ny, nx] == 0:
                                stack.append((ny, nx))

                if len(region) > 50:
                    regions.append(region)
                label += 1

    if regions:
        img_with_bbox = img_with.copy()
        all_regions_bbox = []

        for region in regions:
            ys, xs = zip(*region)
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            grinch_roi = img_with[y_min:y_max + 1, x_min:x_max + 1]
            cv2.imwrite(output_cutout_path, grinch_roi)

            transparent_image = np.zeros((grinch_roi.shape[0], grinch_roi.shape[1], 4), dtype=np.uint8)

            for row in range(grinch_roi.shape[0]):
                for col in range(grinch_roi.shape[1]):
                    if (row + y_min, col + x_min) in region:
                        transparent_image[row, col, :3] = grinch_roi[row, col]
                        transparent_image[row, col, 3] = 255

            cv2.imwrite(output_bez_tla_path, transparent_image)

            cv2.rectangle(img_with_bbox, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            all_regions_bbox.append((x_min, y_min, x_max, y_max))

        cv2.imwrite(output_bbox_path, img_with_bbox)
    else:
        print("Nie znaleziono Grincha na obrazie.")

if __name__ == "__main__":
    image_with_grinches = "edited.jpg"
    image_without_grinches = "org.jpg"
    output_cutout_path = "largest_grinch.png"
    output_bez_tla_path = "bez_tla.png"
    output_bbox_path = "grinch_with_bbox.png"

    find_and_remove_largest_grinch(image_with_grinches, image_without_grinches, output_cutout_path,
                                   output_bez_tla_path, output_bbox_path)
