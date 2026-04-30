from flask import Flask, render_template, request
import os
import uuid
from werkzeug.utils import secure_filename
from utils.inference import run_inference, check_solar

app = Flask(__name__)

UPLOAD_FOLDER = "static/temp"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DEFECT_LABELS = {
    0: "Clean",
    1: "Dusty",
    2: "Bird Droppings",
    3: "Electrical Damage",
    4: "Physical Damage"
}


# ===============================
# VOLTAGE CALCULATION
# ===============================
def compute_voltage(power):
    Vmp = 17.2
    voltage_loss = power * Vmp
    effective_voltage = Vmp - voltage_loss
    return voltage_loss, effective_voltage


# ===============================
# ROUTES
# ===============================
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/preview", methods=["POST"])
def preview():

    file = request.files.get("image")

    if not file or file.filename == "":
        return "No file selected"

    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    save_path = os.path.join(UPLOAD_FOLDER, unique_name)

    file.save(save_path)

    return render_template("starter-page.html", image_name=unique_name)


@app.route("/predict", methods=["POST"])
def predict():

    image_name = request.form.get("image_name")
    image_path = os.path.join(UPLOAD_FOLDER, image_name)

    if not os.path.exists(image_path):
        return "Image not found"
    

    solar_status, prob = check_solar(image_path)
    

    if solar_status == "Not Solar":
        return render_template(
        "starter-page.html",
        image_name=image_name,
        prob=prob,
        error_message="❌ Uploaded image is NOT a solar panel"
    )

    if solar_status == "Uncertain":
        return render_template(
        "starter-page.html",
        image_name=image_name,
        prob=prob,
        error_message="⚠️ Unable to confirm if this is a solar panel"
    )

    

    # ✅ main inference
    power, area_ratio, defect_code, defects = run_inference(image_path)

    refined_pct = power * 100
    area_percent = area_ratio * 100

    defect_name = DEFECT_LABELS.get(defect_code, "Unknown")

    # ✅ voltage
    voltage_loss, effective_voltage = compute_voltage(power)

    return render_template(
        "result.html",
        image_url=os.path.join("static", "temp", image_name),

        refined_pct=round(refined_pct, 2),
        voltage_loss=round(voltage_loss, 2),
        effective_voltage=round(effective_voltage, 2),

        defect_name=defect_name,
        total_area=round(area_percent, 2),
        solar_status=solar_status,
        prob=prob
    )


if __name__ == "__main__":
    app.run(debug=True)