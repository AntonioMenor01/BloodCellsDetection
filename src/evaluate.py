# src/evaluate.py

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model
from src.data_loader import get_data_generators
import numpy as np
import os

def evaluate_model(model_path="models/mobilenet_model.h5", batch_size=16):
    """
    Evalúa el modelo en el test set y genera métricas y gráficas.
    """

    # Cargar generadores
    _, _, test_gen = get_data_generators(batch_size=batch_size)  # solo test

    # Cargar modelo entrenado
    model = load_model(model_path)

    # Evaluar
    loss, acc = model.evaluate(test_gen)
    print(f"Test Accuracy: {acc*100:.2f}%")
    print(f"Test Loss: {loss:.4f}")

    # Predicciones y matriz de confusión
    test_gen.reset()  # asegurarse de empezar desde el principio
    preds = model.predict(test_gen)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_gen.classes

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=list(test_gen.class_indices.keys()))
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Matriz de Confusión")
    plt.savefig("results/confusion_matrix.png")

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    evaluate_model()