import pandas as pd
import tensorflow as tf
from keras.optimizers import Adam
from keras.optimizers.schedules import CosineDecay
from keras.callbacks import EarlyStopping, CSVLogger, LearningRateScheduler
from utils.create_directory import create_directory
from utils.prepare_dataset import prepare_dataset
from utils.tl_model import get_model



def get_callbacks(model_save_dir, model_name, epochs, warmup_epochs):

    early_stopping = EarlyStopping(monitor='val_loss', patience=15, verbose=1, mode='auto', restore_best_weights=True)

    log_csv = f'{model_save_dir}/{model_name}/{model_name}-log.csv'
    csv_logger = CSVLogger(log_csv, separator=",", append=False)

    # scheduler = CosineDecay(initial_learning_rate=0, warmup_steps=warmup_epochs, warmup_target=0.001, decay_steps=(epochs-warmup_epochs))
    # scheduler = LearningRateScheduler(scheduler)

    # return [early_stopping, csv_logger, scheduler]

    return [early_stopping, csv_logger]



def train_model(batch_size, epochs, warmup_epochs, train_labels_csv, test_labels_csv, model_save_dir, model_name):

    create_directory(f'{model_save_dir}/{model_name}')

    train_df = pd.read_csv(train_labels_csv)
    val_df = pd.read_csv(test_labels_csv)

    train_ds = prepare_dataset(train_df, batch_size)
    val_ds = prepare_dataset(val_df, batch_size)

    model = get_model()

    model.compile(optimizer=Adam(1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    callbacks = get_callbacks(model_save_dir, model_name, epochs, warmup_epochs)

    model.fit(x=train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=1)

    model.save(f'{model_save_dir}/{model_name}/{model_name}.keras')



if __name__ == '__main__':

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    batch_size = 32
    epochs = 100
    warmup_epochs = 15

    train_labels_csv = r'src/labels/labels-train.csv'
    test_labels_csv = r'src/labels/labels-test.csv'

    model_save_dir = r'saved-models'
    model_name = "bcd-a1"

    train_model(batch_size, epochs, warmup_epochs, train_labels_csv, test_labels_csv, model_save_dir, model_name)
