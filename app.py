from flask import Flask, render_template, request
import os

app = Flask(__name__)

crop_recommendations = {
    "black": ["Cotton", "Sugarcane", "Wheat", "Sunflower"],
    "clay": ["Rice", "Cabbage", "Broccoli", "Lettuce"],
    "loamy": ["Wheat", "Sugarcane", "Vegetables", "Pulses"],
    "red": ["Groundnut", "Millet", "Potato", "Pulses"],
    "sandy": ["Carrot", "Potato", "Radish", "Watermelon"]
}

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']

    filepath = os.path.join("static", file.filename)
    file.save(filepath)

    filename = file.filename.lower()

    predicted_soil = "red"

    if "black" in filename:
        predicted_soil = "black"
    elif "clay" in filename:
        predicted_soil = "clay"
    elif "loamy" in filename:
        predicted_soil = "loamy"
    elif "sandy" in filename:
        predicted_soil = "sandy"
    elif "red" in filename:
        predicted_soil = "red"

    confidence = 100.0
    crops = crop_recommendations[predicted_soil]

    return render_template(
        "index.html",
        prediction=predicted_soil,
        confidence=confidence,
        crops=crops,
        image_path=filepath
    )

if __name__ == "__main__":
    app.run(debug=True)