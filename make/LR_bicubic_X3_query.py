import os
from PIL import Image

def resize_images(input_folder, output_folder, scale_factor):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Read the image
            image_path = os.path.join(input_folder, filename)
            image = Image.open(image_path)

            # Calculate the new size
            width, height = image.size
            new_width = width * scale_factor
            new_height = height * scale_factor

            # Resize the image using bicubic interpolation
            resized_image = image.resize((new_width, new_height), resample=Image.BICUBIC)

            # Save the resized image with the same filename in the output folder
            output_path = os.path.join(output_folder, filename)
            resized_image.save(output_path)

            print(f"Processed and saved: {output_path}")

input_folder = r"E:\work\SR-YOLO\Functional_module_example\dataset_make\input_image\query"
output_folder = r"E:\work\SR-YOLO\Functional_module_example\dataset_make\output_image\LR_bicubic_X3_query"
scale_factor = 3

resize_images(input_folder, output_folder, scale_factor)
