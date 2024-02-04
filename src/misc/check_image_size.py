import pandas as pd
from skimage import io
from utils.save_image import save_image



def check_image_size(labels_csv, save_dir):

    labels_csv_df = pd.read_csv(labels_csv)

    for index in range(0, len(labels_csv_df)):

        image_dir = labels_csv_df['dir'][index]
        image_name = labels_csv_df['image'][index]

        image = io.imread(f"{image_dir}/{image_name}")

        if (image.shape != (50, 50, 3)):
            save_image(image, image_name, save_dir)



if __name__ == '__main__':

    labels_csv = r'src/labels/labels.csv'

    save_dir = r'not50x50'

    check_image_size(labels_csv, save_dir)
