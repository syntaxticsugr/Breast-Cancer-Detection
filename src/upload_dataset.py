import os
import time
import pandas as pd
import boto3



def upload_dataset(aws_access_key_id, aws_secret_access_key, region_name, bucket_name, labels_csv):

    s3_client = boto3.client(
        service_name = 's3',
        region_name = region_name,
        aws_access_key_id = aws_access_key_id,
        aws_secret_access_key = aws_secret_access_key
    )

    images_path = pd.read_csv(labels_csv)

    total_time = 0
    items_remaining = len(images_path)
    print(f"\033[A\033[A\n\nDone: 0        Remaining: {items_remaining}        Ellapsed Time: 0        Estimated Remaining Time: --:--:--        \n")

    for index in range(0, len(images_path)):

        start_time = time.time()

        image_dir = images_path['dir'][index]
        image_name = images_path['image'][index]

        fullpath = f'{image_dir}/{image_name}'

        s3_client.upload_file(
            fullpath,
            bucket_name,
            fullpath
        )
        
        total_time += (time.time() - start_time)
        items_remaining -= 1
        print(f"\033[A\033[ADone: {index+1}         Remaining: {items_remaining}        Ellapsed Time: {time.strftime('%H:%M:%S', time.gmtime(total_time))}        Estimated Remaining Time: {time.strftime('%H:%M:%S', time.gmtime((items_remaining*(total_time/(index+1)))))}        \n")



if __name__ == "__main__":

    aws_access_key_id = 'AKIAU6GDXKRCIBUPWMV3'
    aws_secret_access_key = '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'

    region_name = 'ap-southeast-2'
    bucket_name = 'bcd-sugrbucket'

    labels_csv = r'src/labels/labels.csv'

    upload_dataset(aws_access_key_id, aws_secret_access_key, region_name, bucket_name, labels_csv)
