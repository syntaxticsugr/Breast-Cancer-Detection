import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.image as mpimg



def plot_dataset_distribution(csv_path, key):

    color_palette = 'muted'

    df = pd.read_csv(csv_path)
    
    levels = {0: "NON IDC", 1: "IDC"}

    counts = df[key].value_counts().reset_index()
    counts.sort_values(by=key, inplace=True)

    fig, axes = plt.subplots(nrows=1, ncols=1)
    sns.barplot(x=counts[key], y=counts['count'], palette=color_palette, hue=counts[key], ax=axes)

    legend_elements = [Patch(color=sns.color_palette(color_palette)[i], label=f"{level} : {levels[level]}")
                       for i, level in enumerate(counts[key])]
    axes.legend(handles=legend_elements, title="IDC")

    for i, v in enumerate(counts['count']):
        axes.text(i, v+((counts.max()['count'])*0.01), str(v), ha='center', va='bottom')

    axes.set_xlabel("IDC")
    axes.set_ylabel("Count")

    fig.suptitle("Dataset Distribution")
    
    plt.show()
    


def visualize_dataset(csv_path, grid_size):

    df = pd.read_csv(csv_path)

    images_per_class = int(grid_size[0] * (grid_size[1] / 2))

    label_0_images = df[df['idc'] == 0].sample(n=images_per_class)
    label_1_images = df[df['idc'] == 1].sample(n=images_per_class)

    _, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(10, 10))

    for ax in axes.flatten():
        ax.axis('off')

    for i, (_, row) in enumerate(label_0_images.iterrows()):
        img_path = f"{row['dir']}/{row['image']}"
        img = mpimg.imread(img_path)
        axes[i//3, i%3].imshow(img)

    for i, (_, row) in enumerate(label_1_images.iterrows()):
        img_path = f"{row['dir']}/{row['image']}"
        img = mpimg.imread(img_path)
        axes[i//3, i%3+3].imshow(img)

    axes[0, 1].set_title('NON IDC', fontsize=14)
    axes[0, 4].set_title('IDC', fontsize=14)

    plt.tight_layout()
    plt.show()



if __name__ == '__main__':

    csv_path = r'src/labels/labels.csv'
    key = 'idc'

    plot_dataset_distribution(csv_path, key)

    visualize_dataset(csv_path, (6 ,6))
