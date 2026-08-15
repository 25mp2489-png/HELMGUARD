from flask import Flask, render_template, request, send_from_directory
from ultralytics import YOLO
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "runs/helmgurad_results"
MODEL_PATH = "best.pt"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)



model = YOLO(MODEL_PATH)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["GET", "POST"])
def upload():

    if request.method == "POST":

        
        if "image" not in request.files:
            return "No file was selected."

        file = request.files["image"]

    
        if file.filename == "":
            return "No file was selected."

        filepath = os.path.join(
            app.config["UPLOAD_FOLDER"],
            file.filename
        )

        file.save(filepath)

        print("File uploaded:", filepath)


        results = model.predict(
            source=filepath,
            conf=0.1,
            save=True,
            project="runs",
            name="helmgurad_results",
            exist_ok=True
        )

        detections = []

        for result in results:

            for box in result.boxes:

                class_id = int(box.cls[0])

                confidence = float(box.conf[0])

                class_name = model.names[class_id]

                detections.append({
                    "class": class_name,
                    "confidence": round(confidence * 100, 2)
                })


        filename = os.path.basename(filepath)

        result_path = os.path.join(
            app.config["RESULT_FOLDER"],
            filename
        )

        return render_template(
            "result.html",
            filename=filename,
            detections=detections,
            result_file=filename
        )


    return render_template("upload.html")



@app.route("/results/<filename>")
def results(filename):

    return send_from_directory(
        app.config["RESULT_FOLDER"],
        filename
    )



@app.route("/dashboard")
def dashboard():

    return render_template("dashboard.html")



@app.route("/history")
def history():

    return render_template("history.html")



if __name__ == "__main__":

    app.run(debug=True)