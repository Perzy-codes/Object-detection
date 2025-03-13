from flask import Flask, render_template, request, send_from_directory
from ultralytics import YOLO
import os

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"
model = YOLO("yolov8n.pt")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]
    if file.filename == "":
        return "No file selected", 400

    # Save the uploaded file
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    # Run YOLOv8 inference
    results = model(filepath)
    annotated_image = results[0].plot()

    # Save the annotated image
    output_path = os.path.join(app.config["UPLOAD_FOLDER"], "output.jpg")
    cv2.imwrite(output_path, annotated_image)

    return render_template("index.html", result_image="output.jpg")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)