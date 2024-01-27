import pandas as pd
import tensorflow as tf
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
# from keras.optimizers.schedules import CosineDecay
from keras.applications.efficientnet_v2 import EfficientNetV2M
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from utils.create_directory import create_directory



def get_model():
    base_model = EfficientNetV2M(
        include_top=False,
        weights='imagenet',
        input_shape=(50, 50, 3)
    )

    for layer in base_model.layers:
        layer.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    inputs = base_model.input
    predictions = Dense(2, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=predictions)

    return model



def read_image(image_path, dr):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image)
    return(image, dr)

def normalize(image, dr):
    return image/255, dr

def configure_for_performance(dataset, batch_size):
    dataset = dataset.cache()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

def prepare_dataset(df, batch_size):
    df['full_path'] = df['dir'] + '/' + df['image']
    dataset = tf.data.Dataset.from_tensor_slices((df['full_path'], df['idc']))
    dataset = dataset.map(read_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = configure_for_performance(dataset, batch_size)
    return dataset



def train_model(batch_size, epochs, train_labels_csv, test_labels_csv):

    train_df = pd.read_csv(train_labels_csv)
    val_df = pd.read_csv(test_labels_csv)

    train_ds = prepare_dataset(train_df, batch_size)
    val_ds = prepare_dataset(val_df, batch_size)

    model = get_model()

    optimizer = Adam(learning_rate=1e-4)

    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto', restore_best_weights=True)

    model.fit(x=train_ds, validation_data=val_ds, epochs=epochs, callbacks=[early_stopping], verbose=1)

    model.save(f'{model_save_dir}/{model_name}.keras')



if __name__ == '__main__':

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    batch_size = 32
    epochs = 100
    train_labels_csv = r'src/labels/labels-v2-train.csv'
    test_labels_csv = r'src/labels/labels-v2-test.csv'

    model_save_dir = r'saved-models'
    model_name = "model"

    create_directory(model_save_dir)

    trained_model = train_model(batch_size, epochs, train_labels_csv, test_labels_csv)
