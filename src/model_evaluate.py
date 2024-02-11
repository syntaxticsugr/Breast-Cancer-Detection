import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix



def plot_confusion_matrix(conf_matrix):

    ax = sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues',
                        xticklabels=['IDC (-)', 'IDC (+)'],
                        yticklabels=['IDC (-)', 'IDC (+)'])

    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix', y=-0.1)
    plt.show()



def get_scores(pred_csv, show_confusion_matrix):

    df = pd.read_csv(pred_csv)

    y_true = df['idc']
    y_pred = df['pred']

    print(classification_report(y_true, y_pred))

    if show_confusion_matrix:

        conf_matrix = confusion_matrix(y_true, y_pred)

        plot_confusion_matrix(conf_matrix)



if __name__ == '__main__':

    pred_csv = r'saved-models/bcd-final/bcd-final-pred.csv'

    show_confusion_matrix = True

    get_scores(pred_csv, show_confusion_matrix)
