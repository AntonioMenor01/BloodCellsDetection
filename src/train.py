from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from src.data_loader import get_data_generators
from src.model import build_model
import os

def train_model(model, epochs=30, batch_size=16, model_path="models/mobilenet_model.h5"):
    # Cargar generadores
    train_gen, val_gen, _ = get_data_generators(batch_size=batch_size)

    # Crear carpeta de modelos si no existe
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Callbacks
    checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, verbose=1)
    earlystop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

    # Entrenamiento
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=[checkpoint, earlystop, reduce_lr]
    )

    return history

if __name__ == "__main__":
    model = build_model(input_shape=(224,224,3), num_classes=4)
    history = train_model(model, epochs=30, batch_size=16)