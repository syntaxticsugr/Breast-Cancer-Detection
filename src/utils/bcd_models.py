from keras import Sequential
from keras.models import Model
from keras.regularizers import l2
from keras.src.optimizers.adam import Adam
from keras.applications.efficientnet_v2 import EfficientNetV2S
from keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D



def get_model(learning_rate):

    # return env2s(learning_rate)
    return simp(learning_rate)



def env2s(learning_rate):

    base_model = EfficientNetV2S(
        include_top=False,
        weights='imagenet',
        input_shape=(50, 50, 3),
        pooling='max'
    )

    for layer in base_model.layers:
        layer.trainable = True

    x = base_model.output
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(6e-3))(x)
    x = Dropout(0.5)(x)

    inputs = base_model.input
    predictions = Dense(2, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=predictions)

    model.compile(optimizer=Adam(learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model



def simp(learning_rate):

    model = Sequential(
        [
            Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(50, 50, 3)),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
            MaxPooling2D((2, 2)),
            BatchNormalization(),
            Dropout(0.3),
            Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.3),
            Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
            Flatten(),
            Dense(128, activation='relu', kernel_initializer='he_uniform'),
            BatchNormalization(),
            Dense(64, activation='relu', kernel_initializer='he_uniform'),
            BatchNormalization(),
            Dense(64, activation='relu', kernel_initializer='he_uniform'),
            Dropout(0.3),
            Dense(24, activation='relu', kernel_initializer='he_uniform'),
            Dense(2, activation='softmax'),
        ]
    )

    model.compile(optimizer=Adam(learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model
