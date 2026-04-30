import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from utils.preprocessing import preprocess_image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ===============================
# LOAD MODELS
# ===============================
yolo_model = YOLO("models/best.pt")

power_model = tf.keras.models.load_model(
    "models/power_loss_model.keras",
    compile=False
)

# Feature extractor (required for your model)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(192, 192, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

feature_extractor = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D()
])



#new code added
# ✅ Solar classifier
solar_model = tf.keras.models.load_model(
    "models/solar_classifier.keras",
    compile=False
)


# ===============================
# YOLO DETECTION
# ===============================
def detect_defects(image_path):

    results = yolo_model(image_path)
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return [], 0.0

    img = results[0].orig_img
    h, w = img.shape[:2]
    total_area = w * h

    defects = []
    total_defect_area = 0

    for i in range(len(boxes)):
        box = boxes.xyxy[i].cpu().numpy()
        cls = int(boxes.cls[i].cpu().numpy())
        conf = float(boxes.conf[i].cpu().numpy())

        x1, y1, x2, y2 = box

        area = ((x2 - x1) * (y2 - y1)) / total_area
        total_defect_area += area

        defects.append({
            "class": cls,
            "area": area,
            "confidence": conf
        })

    return defects, total_defect_area


# ===============================
# MAIN INFERENCE
# ===============================
def run_inference(image_path):

    defects, total_area = detect_defects(image_path)

    if len(defects) == 0:
        defect_code = -1
    else:
        defect_code = max(defects, key=lambda x: x["area"])["class"]

    # ✅ preprocess 
    img_tensor = preprocess_image(image_path)

    # ✅ FIX SHAPE 
    if len(img_tensor.shape) == 5:
        img_tensor = tf.squeeze(img_tensor, axis=1)

    # ✅ extract features
    features = feature_extractor.predict(img_tensor, verbose=0)

    # ✅ predict power
    power = power_model.predict(features, verbose=0)[0][0]
    power = float(np.clip(power, 0.0, 1.0))

    return power, total_area, defect_code, defects



def check_solar(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prob = solar_model.predict(img_array, verbose=0)[0][0]

    if prob >= 0.90:
        return "Solar",prob
    elif prob <= 0.30:
        return "Not Solar",prob
    else:
        return "Uncertain",prob