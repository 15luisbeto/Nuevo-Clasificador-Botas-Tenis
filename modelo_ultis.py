from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as vgg16_preprocess
from tensorflow.keras.models import Model
import numpy as np
import joblib

labels_names = ['Boot', 'Shoe']  # Ajusta según tu modelo

# Función para extraer características
def extraccion_caracteristicas(ruta_imagen, model_red, red_preprocess):
    try:
        print(f"📂 Intentando cargar imagen desde: {ruta_imagen}")
        img = image.load_img(ruta_imagen, target_size=(224, 224))
        x = image.img_to_array(img)
        x1 = np.expand_dims(x, axis=0)
        x2 = red_preprocess(x1)
        features = model_red.predict(x2)
        return features.flatten()
    except Exception as e:
        print(f"❌ Error al extraer características: {str(e)}")
        return None

# Función principal de predicción
def prediccion_modelo(ruta_imagen):
    # 🔁 Cargar modelo de VGG16 SOLO al llamar esta función
    model_vgg = VGG16(weights='imagenet')

    # 🔁 Cargar clasificador
    clf_loaded = joblib.load('models/multinomial_nb_classifier.joblib')

    # Extraer características
    features = extraccion_caracteristicas(ruta_imagen, model_vgg, vgg16_preprocess)
    if features is None:
        return {"error": "No se pudieron extraer características"}

    features = np.array(features).reshape(1, -1)
    predicted_probabilities = clf_loaded.predict_proba(features)
    predicted_label = clf_loaded.predict(features)

    predicted_class = predicted_label[0]
    probability = predicted_probabilities[0][predicted_class]

    return {
        "prediccion": labels_names[predicted_class],
        "confianza": f"{probability * 100:.2f}%"
    }


