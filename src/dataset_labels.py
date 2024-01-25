import os
import pandas as pd



def dataset_labels(dataset_dir, labels_dir):

    labels_df_rows = []

    for (_, _, filenames) in os.walk(dataset_dir, topdown=True):

        for filename in filenames:

            idr = 0 if "class0" in filename else 1

            labels_df_rows.append({"image":filename, "idr":idr})

    labels_df = pd.DataFrame(data=labels_df_rows)
    labels_df.to_csv(f'{labels_dir}/labels.csv', index=False)



if __name__ == "__main__":

    dataset_dir = r'dataset'
    labels_dir = r'labels'

    dataset_labels(dataset_dir, labels_dir)
