import shutil



def copy_folder(source_folder, destination_folder):

    shutil.copytree(source_folder, destination_folder)



if __name__ == "__main__":

    source_folder = r'labels/labels-v2'

    destination_folder = r'saved-models/bcd-final/labels-v2'

    copy_folder(source_folder, destination_folder)
