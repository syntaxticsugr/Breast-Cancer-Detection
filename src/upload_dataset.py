import os
import time
import pandas as pd
import boto3



def images_path_df(dataset_dir):

    images_path_df_rows = []

    for (dirpath, _, filenames) in os.walk(dataset_dir, topdown=True):

        for filename in filenames:

            images_path_df_rows.append({"full_path":f'{dirpath}/{filename}'})

    image_paths_df = pd.DataFrame(data=images_path_df_rows)
    return image_paths_df



def upload_dataset(aws_access_key_id, aws_secret_access_key, region_name, bucket_name, dataset_dir, start_index):

    s3_client = boto3.client(
        service_name = 's3',
        region_name = region_name,
        aws_access_key_id = aws_access_key_id,
        aws_secret_access_key = aws_secret_access_key
    )

    images_path = images_path_df(dataset_dir)

    total_time = 0
    items_remaining = (len(images_path)-start_index)
    print(f"\033[A\033[A\n\nDone: 0        Remaining: {items_remaining}        Ellapsed Time: 0        Estimated Remaining Time: --:--:--        \n")

    for index in range(start_index, len(images_path)):

        start_time = time.time()

        image_path = images_path['full_path'][index]
        image_name = os.path.basename(image_path)

        s3_client.upload_file(
            image_path,
            bucket_name,
            f'dataset/kaggle/{image_name}'
        )
        
        total_time += (time.time() - start_time)
        items_remaining -= 1
        print(f"\033[A\033[ADone: {(index+1)-start_index}         Remaining: {items_remaining}        Ellapsed Time: {time.strftime('%H:%M:%S', time.gmtime(total_time))}        Estimated Remaining Time: {time.strftime('%H:%M:%S', time.gmtime((items_remaining*(total_time/((index+1)-start_index)))))}        \n")



if __name__ == "__main__":

    aws_access_key_id = 'AKIAU6GDXKRCIBUPWMV3'
    aws_secret_access_key = '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'

    region_name = 'ap-southeast-2'
    bucket_name = 'bcd-sugrbucket'

    dataset_dir = r'dataset'
    start_index = 0

    upload_dataset(aws_access_key_id, aws_secret_access_key, region_name, bucket_name, dataset_dir, start_index)
