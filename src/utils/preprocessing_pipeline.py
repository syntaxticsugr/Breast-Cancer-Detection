import os
from utils.image_processing.resize_image import resize_image
from utils.image_processing.rotate_image import rotate_image
from utils.image_processing.flip_image import flip_horizontal
from utils.save_image import save_image



def re_ro(image, image_name, output_dir):

    resize_hw = (50, 50)
    rotate_angles = [0, 180]

    file_name, extension = os.path.splitext(image_name)

    resized_image = resize_image(image, resize_hw)

    for angle in rotate_angles:

        rotated_image = rotate_image(resized_image, angle)
        new_name = f'{file_name}_{angle}{extension}'
        save_image(rotated_image, new_name, output_dir)



def re_ro_fh(image, image_name, output_dir):

    resize_hw = (50, 50)
    rotate_angles = [0, 90, 180, 270]

    file_name, extension = os.path.splitext(image_name)

    resized_image = resize_image(image, resize_hw)

    hr_flipped_image = flip_horizontal(resized_image)
    new_name = f'{file_name}_hf{extension}'
    save_image(hr_flipped_image, new_name, output_dir)

    for angle in rotate_angles:

        rotated_image = rotate_image(resized_image, angle)
        new_name = f'{file_name}_{angle}{extension}'
        save_image(rotated_image, new_name, output_dir)
