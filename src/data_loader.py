import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generators(base_dir="data/raw/dataset2-master/images",
                        target_size=(224, 224),
                        batch_size=16):
    """
    Crea y devuelve generadores de datos para train, test_simple y test.
    Con data augmentation para entrenamiento.
    """
    # Comprobar que las carpetas existen
    for folder in ["TRAIN", "TEST_SIMPLE", "TEST"]:
        folder_path = os.path.join(base_dir, folder)
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"No se encuentra la carpeta: {folder_path}")

    # Data augmentation para entrenamiento
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )

    # Generadores
    train_generator = train_datagen.flow_from_directory(
        os.path.join(base_dir, "TRAIN"),
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    # Validación y test: solo normalización
    val_test_datagen = ImageDataGenerator(rescale=1./255)

    test_simple_generator = val_test_datagen.flow_from_directory(
        os.path.join(base_dir, "TEST_SIMPLE"),
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    test_generator = val_test_datagen.flow_from_directory(
        os.path.join(base_dir, "TEST"),
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, test_simple_generator, test_generator