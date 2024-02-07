from keras import layers
from keras import Sequential
from keras.src.optimizers.adam import Adam



def get_model(learning_rate):

    return simp(learning_rate)



def simp(learning_rate):

    model = Sequential(
        [
            layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(50, 50, 3)),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.4),

            layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.4),

            layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.4),

            layers.Flatten(),
            layers.Dense(512, activation='relu', kernel_initializer='he_uniform'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),

            layers.Dense(256, activation='relu', kernel_initializer='he_uniform'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),

            layers.Dense(128, activation='relu', kernel_initializer='he_uniform'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),

            layers.Dense(2, activation='softmax')
        ]
    )

    model.compile(optimizer=Adam(learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model
