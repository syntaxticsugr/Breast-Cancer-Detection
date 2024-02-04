import pandas as pd
import numpy as np
import tensorflow as tf
from keras.optimizers import Adam
# from keras.optimizers.schedules import CosineDecay
from keras.callbacks import EarlyStopping, CSVLogger, LearningRateScheduler
from utils.create_directory import create_directory
from utils.prepare_dataset import prepare_dataset
from utils.tl_model import get_model



class CosineDecayWithWarmup(LearningRateScheduler):
    def __init__(self, learning_rate_base, total_epochs, warmup_epochs, verbose=0):
        super(CosineDecayWithWarmup, self).__init__(self.lr_schedule, verbose)
        self.learning_rate_base = learning_rate_base
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.verbose = verbose

    def lr_schedule(self, epoch):
        if epoch < self.warmup_epochs:
            lr = (self.learning_rate_base / self.warmup_epochs) * (epoch + 1)
        else:
            lr = self.learning_rate_base * 0.5 * (1 + np.cos((epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs) * np.pi))
        return lr



def get_callbacks(model_save_dir, model_name, epochs, warmup_epochs, learning_rate, early_stop_patience):

    early_stopping = EarlyStopping(monitor='val_loss', patience=early_stop_patience, verbose=1, mode='auto', restore_best_weights=True)

    log_csv = f'{model_save_dir}/{model_name}/{model_name}-log.csv'
    csv_logger = CSVLogger(log_csv, separator=",", append=False)

    # scheduler = CosineDecay(initial_learning_rate=0, warmup_steps=warmup_epochs, warmup_target=learning_rate, decay_steps=(epochs-warmup_epochs))
    # scheduler = LearningRateScheduler(scheduler)

    scheduler = CosineDecayWithWarmup(learning_rate, epochs, warmup_epochs, verbose=1)

    return [early_stopping, csv_logger, scheduler]

    # return [early_stopping, csv_logger]



def train_model(batch_size, epochs, warmup_epochs, input_shape, learning_rate, early_stop_patience, train_labels_csv, val_labels_csv, model_save_dir, model_name):

    create_directory(f'{model_save_dir}/{model_name}')

    train_df = pd.read_csv(train_labels_csv)
    val_df = pd.read_csv(val_labels_csv)

    train_ds = prepare_dataset(train_df, batch_size)
    val_ds = prepare_dataset(val_df, batch_size)

    model = get_model(input_shape)

    model.compile(optimizer=Adam(learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    callbacks = get_callbacks(model_save_dir, model_name, epochs, warmup_epochs, learning_rate, early_stop_patience)

    model.fit(x=train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=1)

    model.save(f'{model_save_dir}/{model_name}/{model_name}.keras')



if __name__ == '__main__':

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    train_labels_csv = r'labels/labels-v2-train.csv'
    val_labels_csv = r'labels/labels-v2-val.csv'

    batch_size = 32
    epochs = 20
    warmup_epochs = 5
    input_shape = (50, 50, 3)
    learning_rate = 0.01
    early_stop_patience = 5

    model_save_dir = r'saved-models'
    model_name = "bcd-a1"

    train_model(batch_size, epochs, warmup_epochs, input_shape, learning_rate, early_stop_patience, train_labels_csv, val_labels_csv, model_save_dir, model_name)
