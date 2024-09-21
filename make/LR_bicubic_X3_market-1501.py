import os
import cv2
import numpy as np

def bicubic_interpolation(image, scale):
    height, width = image.shape[:2]
    new_height = int(height / scale)
    new_width = int(width / scale)

    resized_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    for i in range(new_height):
        for j in range(new_width):
            x = i * scale
            y = j * scale

            x_floor = int(x)
            y_floor = int(y)

            dx = x - x_floor
            dy = y - y_floor

            # Calculate the weights for interpolation
            wx = [weight_cubic(dx + 1.0 - k) for k in range(4)]
            wy = [weight_cubic(dy + 1.0 - k) for k in range(4)]

            # Perform bicubic interpolation
            pixel = np.zeros(3, dtype=np.float32)
            for m in range(4):
                for n in range(4):
                    x_index = min(max(x_floor - 1 + m, 0), height - 1)
                    y_index = min(max(y_floor - 1 + n, 0), width - 1)

                    pixel += wx[m] * wy[n] * image[x_index, y_index]

            resized_image[i, j] = pixel.astype(np.uint8)

    return resized_image

def weight_cubic(t):
    a = -0.5
    if abs(t) <= 1:
        return (a + 2) * abs(t) ** 3 - (a + 3) * abs(t) ** 2 + 1
    elif 1 < abs(t) <= 2:
        return a * abs(t) ** 3 - 5 * a * abs(t) ** 2 + 8 * a * abs(t) - 4 * a
    else:
        return 0

input_folder = r"E:\work\SR-YOLO\Functional_module_example\dataset_make\input_image\market-1501\gt_bbox"
output_folder = r"E:\work\SR-YOLO\Functional_module_example\dataset_make\output_image\LR_bicubic_X3_market-1501"

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Process each image in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Read the image
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        # Apply bicubic interpolation
        downsampled_image = bicubic_interpolation(image, 3)

        # Save the processed image with the same filename in the output folder
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, downsampled_image)

        print(f"Processed and saved: {output_path}")
