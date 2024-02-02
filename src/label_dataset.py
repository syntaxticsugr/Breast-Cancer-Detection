import os
import pandas as pd
from utils.create_directory import create_directory
from sklearn.model_selection import train_test_split



def label_dataset(dataset_dir, labels_dir, labels_filename, tt_split):

    create_directory(labels_dir)

    labels_df_rows = []

    for (dirpath, _, filenames) in os.walk(dataset_dir, topdown=True):

        for filename in filenames:

            idc = 0 if "class0" in filename else 1

            labels_df_rows.append({"dir":dirpath, "image":filename, "idc":idc})

    labels_df = pd.DataFrame(data=labels_df_rows)
    labels_df.to_csv(f'{labels_dir}/{labels_filename}.csv', index=False)

    if tt_split:
        train_df, test_df = train_test_split(labels_df, test_size=0.2, shuffle=True)
        train_df.to_csv(f'{labels_dir}/{labels_filename}-train.csv', index=False)
        test_df.to_csv(f'{labels_dir}/{labels_filename}-test.csv', index=False)




if __name__ == "__main__":

    dataset_dir = r'dataset/kaggle/breast-histopathology-images/IDC_regular_ps50_idx5'
    labels_dir = r'src/labels'

    labels_filename = 'labels'

    tt_split = False

    label_dataset(dataset_dir, labels_dir, labels_filename, tt_split)
