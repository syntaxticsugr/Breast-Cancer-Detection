from keras.models import Model
from keras.regularizers import l2
from keras.applications.efficientnet_v2 import EfficientNetV2M
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization



def get_model():

    return env2m()



def env2m():

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
