import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping, CSVLogger, ReduceLROnPlateau
from utils.create_directory import create_directory
from utils.prepare_dataset import prepare_dataset
from utils.bcd_models import get_model



def get_callbacks(model_save_dir, model_name):

    early_stopping = EarlyStopping(patience=10, restore_best_weights=True, verbose=1)

    log_csv = f'{model_save_dir}/{model_name}/{model_name}-fit-log.csv'
    csv_logger = CSVLogger(log_csv, separator=",", append=False)

    scheduler = ReduceLROnPlateau(factor=0.1, patience=5, verbose=1)

    return [early_stopping, csv_logger, scheduler]



def train_model(batch_size, epochs, learning_rate, train_csv, val_csv, model_save_dir, model_name):

    create_directory(f'{model_save_dir}/{model_name}')

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    train_ds = prepare_dataset(train_df, batch_size)
    val_ds = prepare_dataset(val_df, batch_size)

    callbacks = get_callbacks(model_save_dir, model_name)

    model = get_model(learning_rate)

    model.fit(x=train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=1)

    model.save(f'{model_save_dir}/{model_name}/{model_name}.keras')



if __name__ == '__main__':

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    train_csv = r'labels/labels-v2/labels-v2-train.csv'
    val_csv = r'labels/labels-v2/labels-v2-val.csv'

    batch_size = 32
    epochs = 50
    learning_rate = 1e-2

    model_save_dir = r'saved-models'
    model_name = "bcd-final"

    train_model(batch_size, epochs, learning_rate, train_csv, val_csv, model_save_dir, model_name)
