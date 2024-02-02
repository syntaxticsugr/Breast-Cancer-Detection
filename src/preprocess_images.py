import os
import time
import pandas as pd
from skimage import io
from concurrent.futures import ThreadPoolExecutor
from utils.preprocessing_pipeline import preprocessing_pipeline



def preprocess_images(labels_csv, start_index, resize_hw, rotate_angles, output_dir):

    print("\n\n")

    labels_csv_df = pd.read_csv(labels_csv)

    error_images = []

    total_time = 0
    items_remaining = (len(labels_csv_df)-start_index)
    print(f"\033[A\033[AProcessed: 0        Remaining: {items_remaining}        Ellapsed Time: 0        Estimated Remaining Time: --:--:--        \n")

    with ThreadPoolExecutor() as executor:

        for index in range(start_index, len(labels_csv_df)):

            start_time = time.time()

            image_dir = labels_csv_df['dir'][index]
            image_name = labels_csv_df['image'][index]
            idc = labels_csv_df['idc'][index]

            file_name, extension = os.path.splitext(image_name)

            try:
                image = io.imread(f"{image_dir}/{image_name}")

                executor.submit(preprocessing_pipeline, image, idc, file_name, extension, resize_hw, rotate_angles, output_dir)

            except:
                error_images.append(image_name)

            total_time += (time.time() - start_time)
            items_remaining -= 1
            print(f"\033[A\033[AProcessed: {(index+1)-start_index}        Remaining: {items_remaining}        Ellapsed Time: {time.strftime('%H:%M:%S', time.gmtime(total_time))}        Estimated Remaining Time: {time.strftime('%H:%M:%S', time.gmtime((items_remaining*(total_time/((index+1)-start_index)))))}        \n")

        print(error_images)



if __name__ == '__main__':

    # Required Paths
    labels_csv = r'src/labels/labels.csv'
    output_dir = r'dataset/re-50x50-ro-4x2'

    # Start Index To PreProcess Images From
    # Useful In Case When Need To Resume Preprocessing From Certain Index
    start_index = 0

    resize_hw = (50, 50)
    # idc: angles
    rotate_angles = {
        0: [0, 180],
        1: [0, 90, 180, 270]
    }

    preprocess_images(labels_csv, start_index, resize_hw, rotate_angles, output_dir)
