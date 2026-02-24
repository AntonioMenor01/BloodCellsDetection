import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

MODEL_PATH = "../models/mobilenet_model.h5"
IMG_SIZE = (224, 224)

class_names = ["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"]

# Cargar modelo
model = load_model(MODEL_PATH)


def predict_image(img_path):

    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    predictions = model.predict(img_array)[0]

    pred_class = np.argmax(predictions)
    confidence = np.max(predictions)

    return img, predictions, class_names[pred_class], confidence


def show_results(img, predictions, label, confidence):

    plt.imshow(img)
    plt.axis("off")

    plt.title(f"Predicción: {label} ({confidence:.2f})")
    plt.show()

    print("\nProbabilidades por clase:")
    for i, prob in enumerate(predictions):
        print(f"{class_names[i]}: {prob:.4f}")


if __name__ == "__main__":

    img_path = "../data/test_images/Eosinophils_prueba.jpg"

    img, preds, label, conf = predict_image(img_path)

    show_results(img, preds, label, conf)