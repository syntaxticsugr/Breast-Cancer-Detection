import os
import pandas as pd
from utils.create_directory import create_directory



def label_dataset(dataset_dir, labels_dir, labels_filename):

    create_directory(labels_dir)

    labels_df_rows = []

    for (dirpath, _, filenames) in os.walk(dataset_dir, topdown=True):

        for filename in filenames:

            idr = 0 if "class0" in filename else 1

            labels_df_rows.append({"dir":dirpath, "image":filename, "idc":idr})

    labels_df = pd.DataFrame(data=labels_df_rows)
    labels_df.to_csv(f'{labels_dir}/{labels_filename}.csv', index=False)



if __name__ == "__main__":

    dataset_dir = r'dataset/kaggle/breast-histopathology-images/IDC_regular_ps50_idx5'
    labels_dir = r'src/labels'

    labels_filename = 'labels'

    label_dataset(dataset_dir, labels_dir, labels_filename)
