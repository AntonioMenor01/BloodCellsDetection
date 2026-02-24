from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

def build_model(input_shape=(224, 224, 3), num_classes=4):
    """
    Construye el modelo MobileNetV2 para clasificación de células con fine-tuning.
    """
    # Cargar base MobileNetV2 preentrenada sin las capas superiores
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)

    # Fine-tuning: congelar capas bajas, dejar entrenables las últimas 50
    base_model.trainable = True
    for layer in base_model.layers[:-50]:
        layer.trainable = False

    # Añadir capas personalizadas
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.001))(x)

    # Crear modelo final
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compilar
    model.compile(optimizer=Adam(learning_rate=1e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model