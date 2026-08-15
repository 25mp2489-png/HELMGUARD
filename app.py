from flask import Flask, render_template, request, send_from_directory
from ultralytics import YOLO
from datetime import datetime, timedelta
import sqlite3
import os

app = Flask(__name__)

# =========================================================
# SETTINGS
# =========================================================

UPLOAD_FOLDER = "uploads"
MODEL_PATH = "best.pt"
DATABASE = "helmgurad.db"

DEFAULT_LOCATION = "RIT Kottayam, Kerala"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# =========================================================
# LOAD YOLO MODEL
# =========================================================

model = YOLO(MODEL_PATH)


# =========================================================
# DATABASE INITIALIZATION
# =========================================================

def init_database():

    conn = sqlite3.connect(DATABASE)

    cursor = conn.cursor()

    # -----------------------------------------------------
    # No-Helmet violations
    # -----------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS violations (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            image TEXT,

            date TEXT,

            time TEXT,

            location TEXT,

            detection_type TEXT,

            confidence REAL

        )
    """)

    # -----------------------------------------------------
    # ALL detections
    # -----------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detections (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            image TEXT,

            date TEXT,

            time TEXT,

            location TEXT,

            detection_type TEXT,

            confidence REAL

        )
    """)

    conn.commit()
    conn.close()

    print("Database initialized successfully.")


# =========================================================
# SAVE ALL DETECTIONS
# =========================================================

def save_detection(
    image,
    date,
    time,
    location,
    detection_type,
    confidence
):

    conn = sqlite3.connect(DATABASE)

    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO detections
        (
            image,
            date,
            time,
            location,
            detection_type,
            confidence
        )
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        image,
        date,
        time,
        location,
        detection_type,
        confidence
    ))

    conn.commit()
    conn.close()


# =========================================================
# SAVE NO-HELMET VIOLATION
# =========================================================

def save_violation(
    image,
    date,
    time,
    location,
    detection_type,
    confidence
):

    conn = sqlite3.connect(DATABASE)

    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO violations
        (
            image,
            date,
            time,
            location,
            detection_type,
            confidence
        )
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        image,
        date,
        time,
        location,
        detection_type,
        confidence
    ))

    conn.commit()
    conn.close()

    print("Violation saved to database.")


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    return render_template("index.html")


# =========================================================
# UPLOAD + YOLO DETECTION
# =========================================================

@app.route("/upload", methods=["GET", "POST"])
def upload():

    if request.method == "POST":

        # -------------------------------------------------
        # Check uploaded file
        # -------------------------------------------------

        if "image" not in request.files:

            return "No file was selected."

        file = request.files["image"]

        if file.filename == "":

            return "No file was selected."

        # -------------------------------------------------
        # Filename
        # -------------------------------------------------

        filename = os.path.basename(file.filename)

        # -------------------------------------------------
        # Save uploaded file
        # -------------------------------------------------

        filepath = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )

        file.save(filepath)

        print("File uploaded:", filepath)

        # -------------------------------------------------
        # Create unique YOLO folder
        # -------------------------------------------------

        run_name = (
            "detect_"
            + datetime.now().strftime(
                "%Y%m%d_%H%M%S_%f"
            )
        )

        # -------------------------------------------------
        # YOLO detection
        # -------------------------------------------------

        results = model.predict(

            source=filepath,

            conf=0.1,

            save=True,

            project="runs",

            name=run_name,

            exist_ok=True

        )

        # -------------------------------------------------
        # YOLO output directory
        # -------------------------------------------------

        output_dir = str(results[0].save_dir)

        print("YOLO output directory:")
        print(output_dir)

        # -------------------------------------------------
        # Find result file
        # -------------------------------------------------

        output_file = None

        original_name = os.path.splitext(
            filename
        )[0].lower()

        for file_name in os.listdir(output_dir):

            file_name_without_ext = os.path.splitext(
                file_name
            )[0].lower()

            if file_name_without_ext == original_name:

                output_file = file_name

                break

        # Fallback
        if output_file is None:

            files = os.listdir(output_dir)

            if files:

                output_file = files[0]

        print("Output file:", output_file)

        # =================================================
        # CURRENT DATE & TIME
        # =================================================

        now = datetime.now()

        current_date = now.strftime("%Y-%m-%d")
        current_time = now.strftime("%H:%M:%S")

        # =================================================
        # GET DETECTIONS
        # =================================================

        detections = []

        for result in results:

            if result.boxes is not None:

                for box in result.boxes:

                    class_id = int(box.cls[0])

                    confidence = float(box.conf[0])

                    class_name = model.names[class_id]

                    confidence_percent = round(
                        confidence * 100,
                        2
                    )

                    detections.append({

                        "class": class_name,

                        "confidence":
                            confidence_percent

                    })

                    # =====================================
                    # SAVE EVERY DETECTION
                    # =====================================

                    save_detection(

                        image=output_file,

                        date=current_date,

                        time=current_time,

                        location=DEFAULT_LOCATION,

                        detection_type=class_name,

                        confidence=confidence_percent

                    )

        # =================================================
        # COUNT DETECTIONS
        # =================================================

        helmet_count = 0
        nohelmet_count = 0

        for detection in detections:

            class_name = detection["class"].lower()

            if class_name == "helmet":

                helmet_count += 1

            elif class_name in [
                "no-helmet",
                "no helmet"
            ]:

                nohelmet_count += 1

        print(
            "Helmet count:",
            helmet_count
        )

        print(
            "No-Helmet count:",
            nohelmet_count
        )

        # =================================================
        # SAVE NO-HELMET VIOLATION
        # =================================================

        if nohelmet_count > 0:

            for detection in detections:

                if detection["class"].lower() in [
                    "no-helmet",
                    "no helmet"
                ]:

                    save_violation(

                        image=output_file,

                        date=current_date,

                        time=current_time,

                        location=DEFAULT_LOCATION,

                        detection_type="No-Helmet",

                        confidence=detection["confidence"]

                    )

        # =================================================
        # RESULT PAGE
        # =================================================

        return render_template(

            "result.html",

            filename=filename,

            detections=detections,

            helmet_count=helmet_count,

            nohelmet_count=nohelmet_count,

            result_file=output_file

        )

    return render_template("upload.html")


# =========================================================
# SERVE YOLO RESULT
# =========================================================

@app.route("/results/<path:filename>")
def results(filename):

    base_folder = os.path.join(
        os.getcwd(),
        "runs"
    )

    for root, directories, files in os.walk(base_folder):

        if filename in files:

            return send_from_directory(
                root,
                filename
            )

    return "Result file not found.", 404


# =========================================================
# DASHBOARD
# =========================================================

@app.route("/dashboard")
def dashboard():

    conn = sqlite3.connect(DATABASE)

    cursor = conn.cursor()

    # -----------------------------------------------------
    # Total Helmet
    # -----------------------------------------------------

    cursor.execute("""
        SELECT COUNT(*)
        FROM detections
        WHERE LOWER(detection_type) = 'helmet'
    """)

    total_helmet = cursor.fetchone()[0]

    # -----------------------------------------------------
    # Total No-Helmet
    # -----------------------------------------------------

    cursor.execute("""
        SELECT COUNT(*)
        FROM detections
        WHERE LOWER(detection_type)
        IN ('no-helmet', 'no helmet')
    """)

    total_nohelmet = cursor.fetchone()[0]

    # -----------------------------------------------------
    # Total detections
    # -----------------------------------------------------

    total_detections = (
        total_helmet +
        total_nohelmet
    )

    # -----------------------------------------------------
    # Last 7 days
    # -----------------------------------------------------

    today = datetime.now().date()

    weekly_data = []

    for i in range(6, -1, -1):

        day = today - timedelta(days=i)

        date_string = day.strftime(
            "%Y-%m-%d"
        )

        day_name = day.strftime("%a")

        # Helmet count
        cursor.execute("""
            SELECT COUNT(*)
            FROM detections
            WHERE date = ?
            AND LOWER(detection_type) = 'helmet'
        """, (date_string,))

        helmet = cursor.fetchone()[0]

        # No helmet count
        cursor.execute("""
            SELECT COUNT(*)
            FROM detections
            WHERE date = ?
            AND LOWER(detection_type)
            IN ('no-helmet', 'no helmet')
        """, (date_string,))

        nohelmet = cursor.fetchone()[0]

        weekly_data.append({

            "day": day_name,

            "date": date_string,

            "helmet": helmet,

            "nohelmet": nohelmet

        })

    conn.close()

    return render_template(

        "dashboard.html",

        total_helmet=total_helmet,

        total_nohelmet=total_nohelmet,

        total_detections=total_detections,

        weekly_data=weekly_data

    )


# =========================================================
# HISTORY
# =========================================================

@app.route("/history")
def history():

    conn = sqlite3.connect(DATABASE)

    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM violations
        ORDER BY id DESC
    """)

    violations = cursor.fetchall()

    conn.close()

    return render_template(

        "history.html",

        violations=violations

    )


# =========================================================
# RUN FLASK
# =========================================================

if __name__ == "__main__":

    init_database()

    app.run(
        debug=True
    )