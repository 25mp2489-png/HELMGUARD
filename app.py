from flask import Flask, render_template, request
import os

app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Create uploads folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Upload page
@app.route("/upload", methods=["GET", "POST"])
def upload():

    if request.method == "POST":

        # Check whether a file was submitted
        if "image" not in request.files:
            return "No image was selected."

        image = request.files["image"]

        # Check whether a filename exists
        if image.filename == "":
            return "No image was selected."

        # Save the uploaded image
        filepath = os.path.join(
            app.config["UPLOAD_FOLDER"],
            image.filename
        )

        image.save(filepath)

        return f"Image uploaded successfully: {image.filename}"

    # Display upload page
    return render_template("upload.html")


# Dashboard
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


# History
@app.route("/history")
def history():
    return render_template("history.html")


# Run application
if __name__ == "__main__":
    app.run(debug=True)