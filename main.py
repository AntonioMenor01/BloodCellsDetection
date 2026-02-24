# main.py
import os
from src.model import build_model
from src.train import train_model
from src.evaluate import evaluate_model

def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    model_path = "models/mobilenet_model.h5"

    # Si el modelo no existe, entrenar
    if not os.path.exists(model_path):
        print("Modelo no encontrado, entrenando...")
        model = build_model(input_shape=(224,224,3), num_classes=4)
        train_model(model, epochs=20, batch_size=16, model_path=model_path)
    else:
        print("Modelo encontrado, saltando entrenamiento.")

    # Evaluar
    print("Evaluando el modelo en el test set...")
    evaluate_model(model_path=model_path, batch_size=16)
    print("Pipeline completado ")

if __name__ == "__main__":
    main()