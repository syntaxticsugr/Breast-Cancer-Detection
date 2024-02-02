from utils.image_processing.resize_image import resize_image
from utils.image_processing.rotate_image import rotate_image
from utils.save_image import save_image



def preprocessing_pipeline(image, idc, file_name, extension, resize_hw, rotate_angles, output_dir):

    resized_image = resize_image(image, resize_hw)

    for angle in rotate_angles[idc]:

        rotated_image = rotate_image(resized_image, angle)

        new_name = f'{file_name}_{angle}{extension}'

        save_image(rotated_image, new_name, output_dir)
