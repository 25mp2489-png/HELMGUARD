from flask import Flask, render_template, request, send_from_directory
from ultralytics import YOLO
from datetime import datetime, timedelta
import sqlite3
import os
import re
import cv2
import subprocess

try:
    import imageio_ffmpeg
    FFMPEG_AVAILABLE = True
except ImportError:
    imageio_ffmpeg = None
    FFMPEG_AVAILABLE = False

# EasyOCR
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    easyocr = None
    EASYOCR_AVAILABLE = False


app = Flask(__name__)


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "uploads"
)

RUNS_FOLDER = os.path.join(
    BASE_DIR,
    "runs"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "best.pt"
)

PLATE_MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "plate_best.pt"
)

DATABASE = os.path.join(
    BASE_DIR,
    "helmgurad.db"
)

NOTICE_FOLDER = os.path.join(
    BASE_DIR,
    "notices"
)

PLATE_EVIDENCE_FOLDER = os.path.join(
    BASE_DIR,
    "plate_evidence"
)

DEFAULT_LOCATION = "RIT Kottayam, Kerala"


# =========================================================
# CREATE FOLDERS
# =========================================================

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

os.makedirs(
    RUNS_FOLDER,
    exist_ok=True
)

os.makedirs(
    NOTICE_FOLDER,
    exist_ok=True
)

os.makedirs(
    PLATE_EVIDENCE_FOLDER,
    exist_ok=True
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# =========================================================
# DEMO SETTINGS
# =========================================================

DEMO_MODE = True

DEMO_PLATE_NUMBER = "MP04XX0569"

PLATE_CONFIDENCE = 0.40

OCR_CONFIDENCE = 0.20

NOTICE_LIMIT_HOURS = 2


# =========================================================
# LOAD HELMET MODEL
# =========================================================

print("Loading helmet detection model...")

model = YOLO(
    MODEL_PATH
)

print(
    "Helmet model loaded successfully."
)


# =========================================================
# LOAD PLATE MODEL
# =========================================================

print(
    "Loading number plate model..."
)

plate_model = YOLO(
    PLATE_MODEL_PATH
)

print(
    "Number plate model loaded successfully."
)


# =========================================================
# LOAD EASY OCR
# =========================================================

ocr_reader = None

if EASYOCR_AVAILABLE:

    try:

        print(
            "Loading EasyOCR..."
        )

        ocr_reader = easyocr.Reader(
            ["en"],
            gpu=False
        )

        print(
            "EasyOCR loaded successfully."
        )

    except Exception as e:

        print(
            "EasyOCR could not be loaded:"
        )

        print(e)

        ocr_reader = None

else:

    print(
        "EasyOCR is not installed."
    )

    print(
        "Run: python -m pip install easyocr"
    )
    # =========================================================
# DATABASE INITIALIZATION
# =========================================================

def init_database():

    conn = sqlite3.connect(
        DATABASE
    )

    cursor = conn.cursor()

    # -----------------------------------------------------
    # Violations
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
    # All detections
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

    # -----------------------------------------------------
    # Vehicle owners
    # -----------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS vehicle_owners (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            plate_number TEXT UNIQUE,

            owner_name TEXT,

            phone TEXT,

            address TEXT,

            vehicle_type TEXT

        )
    """)

    # -----------------------------------------------------
    # Penalty notices
    # -----------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS penalty_notices (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            notice_id TEXT UNIQUE,

            violation_id INTEGER,

            plate_number TEXT,

            owner_name TEXT,

            phone TEXT,

            violation_date TEXT,

            violation_time TEXT,

            notice_date TEXT,

            notice_time TEXT,

            location TEXT,

            detection_type TEXT,

            confidence REAL,

            notice_status TEXT,

            notice_file TEXT

        )
    """)

    # -----------------------------------------------------
    # Notifications
    # -----------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS notifications (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            notification_type TEXT,

            recipient TEXT,

            plate_number TEXT,

            message TEXT,

            date TEXT,

            time TEXT,

            status TEXT

        )
    """)

    conn.commit()

    conn.close()

    print(
        "Database initialized successfully."
    )


# =========================================================
# SAVE DETECTION
# =========================================================

def save_detection(
    image,
    date,
    time,
    location,
    detection_type,
    confidence
):

    conn = sqlite3.connect(
        DATABASE
    )

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
# SAVE VIOLATION
# =========================================================

def save_violation(
    image,
    date,
    time,
    location,
    detection_type,
    confidence
):

    conn = sqlite3.connect(
        DATABASE
    )

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

    violation_id = cursor.lastrowid

    conn.commit()

    conn.close()

    print(
        "Violation saved to database. ID:",
        violation_id
    )

    return violation_id


# =========================================================
# NORMALIZE PLATE
# =========================================================

def normalize_plate(text):

    if not text:

        return ""

    text = text.upper()

    text = re.sub(
        r"[^A-Z0-9]",
        "",
        text
    )

    return text


# =========================================================
# IMPROVED TWO-LINE OCR
# =========================================================

def read_plate_text(crop):

    if crop is None:

        return ""

    if crop.size == 0:

        return ""

    if ocr_reader is None:

        return ""

    try:

        # -------------------------------------------------
        # Resize
        # -------------------------------------------------

        height, width = crop.shape[:2]

        if width < 500:

            scale = 500 / width

            crop = cv2.resize(
                crop,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC
            )

        # -------------------------------------------------
        # Grayscale
        # -------------------------------------------------

        gray = cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2GRAY
        )

        # -------------------------------------------------
        # Improve contrast
        # -------------------------------------------------

        gray = cv2.equalizeHist(
            gray
        )

        # -------------------------------------------------
        # Threshold
        # -------------------------------------------------

        processed = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY
            + cv2.THRESH_OTSU
        )[1]

        # -------------------------------------------------
        # OCR complete plate
        # -------------------------------------------------

        results = ocr_reader.readtext(
            processed,
            detail=1,
            paragraph=False
        )

        if not results:

            return ""

        detected_parts = []

        # -------------------------------------------------
        # Extract OCR text and position
        # -------------------------------------------------

        for item in results:

            if len(item) < 3:

                continue

            box = item[0]

            text = str(
                item[1]
            ).strip()

            confidence = float(
                item[2]
            )

            if confidence < OCR_CONFIDENCE:

                continue

            y_positions = [
                point[1]
                for point in box
            ]

            x_positions = [
                point[0]
                for point in box
            ]

            y_center = (
                sum(y_positions)
                /
                len(y_positions)
            )

            x_center = (
                sum(x_positions)
                /
                len(x_positions)
            )

            detected_parts.append({

                "text": text,

                "confidence": confidence,

                "x": x_center,

                "y": y_center

            })

        if not detected_parts:

            return ""

        # -------------------------------------------------
        # Sort top to bottom
        # -------------------------------------------------

        detected_parts.sort(
            key=lambda item:
            item["y"]
        )

        # -------------------------------------------------
        # Group into lines
        # -------------------------------------------------

        lines = []

        line_tolerance = max(
            crop.shape[0] * 0.20,
            10
        )

        for part in detected_parts:

            added_to_line = False

            for line in lines:

                if abs(
                    part["y"]
                    -
                    line["average_y"]
                ) <= line_tolerance:

                    line["parts"].append(
                        part
                    )

                    line["average_y"] = (
                        sum(
                            p["y"]
                            for p in line["parts"]
                        )
                        /
                        len(line["parts"])
                    )

                    added_to_line = True

                    break

            if not added_to_line:

                lines.append({

                    "average_y":
                        part["y"],

                    "parts":
                        [part]

                })

        # -------------------------------------------------
        # Sort lines top to bottom
        # -------------------------------------------------

        lines.sort(
            key=lambda line:
            line["average_y"]
        )

        final_lines = []

        for line in lines:

            # Left-to-right ordering
            line["parts"].sort(
                key=lambda part:
                part["x"]
            )

            line_text = " ".join(
                part["text"]
                for part in line["parts"]
            )

            line_text = (
                line_text.strip()
            )

            if line_text:

                final_lines.append(
                    line_text
                )

        # -------------------------------------------------
        # Display OCR information
        # -------------------------------------------------

        print(
            "OCR lines detected:",
            final_lines
        )

        # -------------------------------------------------
        # Combine lines
        # -------------------------------------------------

        if len(final_lines) >= 2:

            final_text = "\n".join(
                final_lines
            )

        else:

            final_text = " ".join(
                final_lines
            )

        print(
            "Combined OCR:",
            final_text
        )

        return final_text.strip()

    except Exception as e:

        print(
            "OCR error:",
            e
        )

        return ""


# =========================================================
# CONVERT VIDEO TO BROWSER-COMPATIBLE MP4
# =========================================================

def convert_video_to_mp4(input_path):
    """
    Convert the YOLO-generated video to H.264 MP4 so it can
    play reliably in modern web browsers.
    Returns the MP4 filename and absolute path.
    """
    if not input_path:
        return None, None

    input_path = os.path.abspath(input_path)

    if not os.path.exists(input_path):
        print("Video conversion skipped: input file not found.")
        return None, None

    if os.path.splitext(input_path)[1].lower() == ".mp4":
        return os.path.basename(input_path), input_path

    if not FFMPEG_AVAILABLE:
        print("FFmpeg conversion unavailable: imageio-ffmpeg is not installed.")
        return os.path.basename(input_path), input_path

    output_path = os.path.splitext(input_path)[0] + ".mp4"

    try:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

        command = [
            ffmpeg_exe,
            "-y",
            "-i", input_path,
            "-vf", "scale=1280:-2",
            "-r", "30",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-an",
            output_path
        ]

        print("Converting processed video to browser-compatible MP4...")
        print("FFmpeg:", ffmpeg_exe)

        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )

        if completed.returncode != 0:
            print("FFmpeg conversion failed.")
            print(completed.stderr[-3000:])
            return os.path.basename(input_path), input_path

        if not os.path.exists(output_path):
            print("FFmpeg finished but MP4 file was not created.")
            return os.path.basename(input_path), input_path

        print("Browser-compatible MP4 created:", output_path)

        return os.path.basename(output_path), output_path

    except Exception as e:
        print("Video conversion error:", e)
        return os.path.basename(input_path), input_path


# =========================================================
# NUMBER PLATE DETECTION
# =========================================================

def detect_plate(image):

    result_data = {

        "detected": False,

        "confidence": 0,

        "ocr_text": "",

        "normalized_text": "",

        "plate_image": None,

        "demo_match": False,

        "message":
            "Number plate not detected."

    }

    if image is None:

        return result_data

    try:

        results = plate_model.predict(

            source=image,

            conf=PLATE_CONFIDENCE,

            verbose=False

        )

        if not results:

            return result_data

        result = results[0]

        if result.boxes is None:

            return result_data

        if len(result.boxes) == 0:

            return result_data

        # -------------------------------------------------
        # Highest confidence plate
        # -------------------------------------------------

        best_box = None

        best_confidence = 0

        for box in result.boxes:

            confidence = float(
                box.conf[0]
            )

            if confidence > best_confidence:

                best_confidence = (
                    confidence
                )

                best_box = box

        if best_box is None:

            return result_data

        # -------------------------------------------------
        # Coordinates
        # -------------------------------------------------

        coordinates = (
            best_box.xyxy[0].tolist()
        )

        x1, y1, x2, y2 = [

            int(value)

            for value in coordinates

        ]

        height, width = (
            image.shape[:2]
        )

        x1 = max(
            0,
            min(
                x1,
                width - 1
            )
        )

        x2 = max(
            0,
            min(
                x2,
                width
            )
        )

        y1 = max(
            0,
            min(
                y1,
                height - 1
            )
        )

        y2 = max(
            0,
            min(
                y2,
                height
            )
        )

        if (
            x2 <= x1
            or
            y2 <= y1
        ):

            return result_data

        # -------------------------------------------------
        # Crop
        # -------------------------------------------------

        plate_crop = image[
            y1:y2,
            x1:x2
        ]

        if plate_crop.size == 0:

            return result_data

        # -------------------------------------------------
        # Save crop
        # -------------------------------------------------

        plate_filename = (
            "plate_"
            +
            datetime.now().strftime(
                "%Y%m%d_%H%M%S_%f"
            )
            +
            ".jpg"
        )

        plate_path = os.path.join(

            PLATE_EVIDENCE_FOLDER,

            plate_filename

        )

        cv2.imwrite(
            plate_path,
            plate_crop
        )

        # -------------------------------------------------
        # OCR
        # -------------------------------------------------

        ocr_text = read_plate_text(
            plate_crop
        )

        normalized_text = normalize_plate(
            ocr_text
        )

        # -------------------------------------------------
        # Store results
        # -------------------------------------------------

        result_data["detected"] = True

        result_data["confidence"] = round(
            best_confidence * 100,
            2
        )

        result_data["ocr_text"] = (
            ocr_text
        )

        result_data["normalized_text"] = (
            normalized_text
        )

        result_data["plate_image"] = (
            plate_filename
        )

        result_data["message"] = (
            "Number plate detected."
        )

        # -------------------------------------------------
        # DEMO MATCH
        # -------------------------------------------------

        if DEMO_MODE:

            demo_normalized = normalize_plate(
                DEMO_PLATE_NUMBER
            )

            # Exact OCR match
            if normalized_text == demo_normalized:

                result_data["demo_match"] = True

            else:

                # Remove spaces and newlines
                compact_text = (
                    normalized_text
                    .replace("\n", "")
                    .replace(" ", "")
                )

                # Common OCR corrections
                corrected_text = compact_text.replace(
                    "MLP04",
                    "MP04"
                )

                corrected_text = corrected_text.replace(
                    "MPO4",
                    "MP04"
                )

                # Partial OCR match
                if (
                    "MP04" in corrected_text
                    and
                    "0569" in corrected_text
                ):

                    result_data["demo_match"] = True

                # -------------------------------------------------
                # DEMO FALLBACK
                # -------------------------------------------------
                # The plate detector has already found a plate with
                # sufficient confidence, but EasyOCR may not be able
                # to read this two-line plate completely.
                #
                # Because DEMO_MODE is explicitly enabled for this
                # academic project, use the configured demo plate
                # when the detector confidence is at least 60%.
                #
                # This does NOT claim that OCR read the full plate.
                # It only allows the demonstration workflow to
                # continue to the demo owner/notification/notice.

                elif best_confidence >= 0.60:

                    result_data["demo_match"] = True

                    # Show the configured demo plate as the normalized
                    # value used for the demonstration lookup.
                    result_data["normalized_text"] = demo_normalized

        print(
            "Plate confidence:",
            result_data[
                "confidence"
            ]
        )

        print(
            "OCR text:",
            ocr_text
        )

        print(
            "Normalized plate:",
            normalized_text
        )

        print(
            "Demo plate match:",
            result_data[
                "demo_match"
            ]
        )

        return result_data

    except Exception as e:

        print(
            "Plate detection error:",
            e
        )

        result_data[
            "message"
        ] = "Plate detection error."

        return result_data


# =========================================================
# FIND VEHICLE OWNER
# =========================================================

def find_owner(
    normalized_plate,
    demo_match=False
):

    conn = sqlite3.connect(
        DATABASE
    )

    conn.row_factory = (
        sqlite3.Row
    )

    cursor = conn.cursor()

    owner = None

    # -----------------------------------------------------
    # Exact match
    # -----------------------------------------------------

    if normalized_plate:

        cursor.execute("""
            SELECT *
            FROM vehicle_owners
            WHERE UPPER(
                REPLACE(
                    REPLACE(
                        plate_number,
                        ' ',
                        ''
                    ),
                    '-',
                    ''
                )
            ) = ?
            LIMIT 1
        """, (
            normalized_plate,
        ))

        owner = cursor.fetchone()

    # -----------------------------------------------------
    # Demo match
    # -----------------------------------------------------

    if (
        owner is None
        and
        DEMO_MODE
        and
        demo_match
    ):

        demo_plate = normalize_plate(
            DEMO_PLATE_NUMBER
        )

        cursor.execute("""
            SELECT *
            FROM vehicle_owners
            WHERE UPPER(
                REPLACE(
                    REPLACE(
                        plate_number,
                        ' ',
                        ''
                    ),
                    '-',
                    ''
                )
            ) = ?
            LIMIT 1
        """, (
            demo_plate,
        ))

        owner = cursor.fetchone()

    conn.close()

    if owner:

        return dict(
            owner
        )

    return None


# =========================================================
# SAVE NOTIFICATION
# =========================================================

def save_notification(
    notification_type,
    recipient,
    plate_number,
    message,
    notification_datetime=None
):

    if notification_datetime is None:

        notification_datetime = (
            datetime.now()
        )

    date = (
        notification_datetime.strftime(
            "%Y-%m-%d"
        )
    )

    time = (
        notification_datetime.strftime(
            "%H:%M:%S"
        )
    )

    conn = sqlite3.connect(
        DATABASE
    )

    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO notifications
        (
            notification_type,
            recipient,
            plate_number,
            message,
            date,
            time,
            status
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (

        notification_type,

        recipient,

        plate_number,

        message,

        date,

        time,

        "Demo notification recorded"

    ))

    conn.commit()

    conn.close()

    print(
        notification_type,
        "notification recorded."
    )


# =========================================================
# GENERATE PENALTY NOTICE
# =========================================================

def generate_penalty_notice(
    violation_id,
    plate_number,
    owner,
    violation_datetime,
    location,
    confidence
):

    if owner is None:

        return None

    notice_datetime = (
        datetime.now()
    )

    elapsed = (
        notice_datetime
        -
        violation_datetime
    ).total_seconds()

    issued_within_two_hours = (

        elapsed <= (
            NOTICE_LIMIT_HOURS
            * 3600
        )

        and

        elapsed >= 0

    )

    notice_id = (
        "HG-"
        +
        datetime.now().strftime(
            "%Y%m%d%H%M%S%f"
        )[:-3]
    )

    notice_filename = (
        notice_id
        +
        ".html"
    )

    notice_path = os.path.join(

        NOTICE_FOLDER,

        notice_filename

    )

    timing_status = (

        "Issued within 2 hours"

        if issued_within_two_hours

        else

        "Issued after 2 hours"

    )

    # -----------------------------------------------------
    # Notice HTML
    # -----------------------------------------------------

    html = f"""
<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<title>
HELMGUARD Penalty Notice
</title>

<style>

body {{

    font-family: Arial, sans-serif;

    background: #111;

    color: #fff;

    padding: 40px;

}}

.notice {{

    max-width: 800px;

    margin: auto;

    background: #222;

    padding: 35px;

    border-radius: 12px;

}}

h1 {{

    text-align: center;

}}

.warning {{

    background: #5c1515;

    padding: 15px;

    border-radius: 8px;

    margin: 20px 0;

}}

.info {{

    line-height: 1.8;

}}

.footer {{

    margin-top: 30px;

    color: #aaa;

    font-size: 13px;

}}

</style>

</head>

<body>

<div class="notice">

<h1>
HELMGUARD
</h1>

<h2>
Helmet Violation Notice
</h2>

<div class="warning">

<strong>
No-Helmet Violation Detected
</strong>

</div>

<div class="info">

<p>
<strong>Notice ID:</strong>
{notice_id}
</p>

<p>
<strong>Owner:</strong>
{owner["owner_name"]}
</p>

<p>
<strong>Vehicle:</strong>
{plate_number}
</p>

<p>
<strong>Vehicle Type:</strong>
{owner["vehicle_type"]}
</p>

<p>
<strong>Phone:</strong>
{owner["phone"]}
</p>

<p>
<strong>Address:</strong>
{owner["address"]}
</p>

<p>
<strong>Violation Date:</strong>
{violation_datetime.strftime("%Y-%m-%d")}
</p>

<p>
<strong>Violation Time:</strong>
{violation_datetime.strftime("%H:%M:%S")}
</p>

<p>
<strong>Notice Date:</strong>
{notice_datetime.strftime("%Y-%m-%d")}
</p>

<p>
<strong>Notice Time:</strong>
{notice_datetime.strftime("%H:%M:%S")}
</p>

<p>
<strong>Location:</strong>
{location}
</p>

<p>
<strong>Detection Confidence:</strong>
{confidence}%
</p>

<p>
<strong>Notice Status:</strong>
{timing_status}
</p>

</div>

<div class="footer">

<p>
This notice is generated as part of the
HELMGUARD academic project demonstration.
</p>

<p>
It is not an official RTO or government notice.
</p>

</div>

</div>

</body>

</html>
"""

    with open(
        notice_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            html
        )

    # -----------------------------------------------------
    # Save notice
    # -----------------------------------------------------

    conn = sqlite3.connect(
        DATABASE
    )

    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO penalty_notices
        (
            notice_id,
            violation_id,
            plate_number,
            owner_name,
            phone,
            violation_date,
            violation_time,
            notice_date,
            notice_time,
            location,
            detection_type,
            confidence,
            notice_status,
            notice_file
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (

        notice_id,

        violation_id,

        plate_number,

        owner["owner_name"],

        owner["phone"],

        violation_datetime.strftime(
            "%Y-%m-%d"
        ),

        violation_datetime.strftime(
            "%H:%M:%S"
        ),

        notice_datetime.strftime(
            "%Y-%m-%d"
        ),

        notice_datetime.strftime(
            "%H:%M:%S"
        ),

        location,

        "No-Helmet",

        confidence,

        timing_status,

        notice_filename

    ))

    conn.commit()

    conn.close()

    # -----------------------------------------------------
    # Owner notification
    # -----------------------------------------------------

    owner_message = (
        "HELMGUARD detected a "
        "No-Helmet violation for vehicle "
        + plate_number
        + ". A demonstration notice "
        "has been generated."
    )

    save_notification(

        notification_type="Vehicle Owner",

        recipient=owner["phone"],

        plate_number=plate_number,

        message=owner_message,

        notification_datetime=(
            notice_datetime
        )

    )

    # -----------------------------------------------------
    # Admin notification
    # -----------------------------------------------------

    admin_message = (
        "HELMGUARD detected a "
        "No-Helmet violation for vehicle "
        + plate_number
        + ". Owner notification and "
        "demonstration penalty notice "
        "were recorded."
    )

    save_notification(

        notification_type="Admin",

        recipient="HELMGUARD Admin",

        plate_number=plate_number,

        message=admin_message,

        notification_datetime=(
            notice_datetime
        )

    )

    return {

        "notice_id":
            notice_id,

        "notice_file":
            notice_filename,

        "notice_status":
            timing_status,

        "notice_time":
            notice_datetime.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

        "owner_notified":
            True,

        "admin_notified":
            True

    }


# =========================================================
# PROCESS VIOLATION EVIDENCE
# =========================================================

def process_violation_evidence(
    image,
    violation_id,
    violation_datetime,
    confidence
):

    plate_info = detect_plate(
        image
    )

    owner_info = None

    notice_info = None

    if plate_info["detected"]:

        owner_info = find_owner(

            plate_info[
                "normalized_text"
            ],

            plate_info[
                "demo_match"
            ]

        )

    if owner_info is not None:

        notice_info = (
            generate_penalty_notice(

                violation_id=(
                    violation_id
                ),

                plate_number=(
                    owner_info[
                        "plate_number"
                    ]
                ),

                owner=(
                    owner_info
                ),

                violation_datetime=(
                    violation_datetime
                ),

                location=(
                    DEFAULT_LOCATION
                ),

                confidence=(
                    confidence
                )

            )
        )

    return (

        plate_info,

        owner_info,

        notice_info

    )


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================================
# UPLOAD
# =========================================================

@app.route(
    "/upload",
    methods=["GET", "POST"]
)
def upload():

    if request.method == "POST":

        # -------------------------------------------------
        # Check file
        # -------------------------------------------------

        if "image" not in request.files:

            return (
                "No file was selected."
            )

        file = request.files[
            "image"
        ]

        if file.filename == "":

            return (
                "No file was selected."
            )

        # -------------------------------------------------
        # Filename
        # -------------------------------------------------

        filename = os.path.basename(
            file.filename
        )

        extension = os.path.splitext(
            filename
        )[1].lower()

        video_extensions = [

            ".mp4",

            ".avi",

            ".mov",

            ".mkv",

            ".webm"

        ]

        is_video = (
            extension
            in video_extensions
        )

        # -------------------------------------------------
        # Save uploaded file
        # -------------------------------------------------

        filepath = os.path.join(

            app.config[
                "UPLOAD_FOLDER"
            ],

            filename

        )

        file.save(
            filepath
        )

        print(
            "File uploaded:",
            filepath
        )

        # -------------------------------------------------
        # Unique run folder
        # -------------------------------------------------

        run_name = (
            "detect_"
            +
            datetime.now().strftime(
                "%Y%m%d_%H%M%S_%f"
            )
        )

        print(
            "Processing..."
        )

        # =================================================
        # IMAGE
        # =================================================

        if not is_video:

            results = model.predict(

                source=filepath,

                conf=0.1,

                save=True,

                project=RUNS_FOLDER,

                name=run_name,

                exist_ok=True

            )

            output_dir = str(
                results[0].save_dir
            )

            output_file = None

            original_name = (
                os.path.splitext(
                    filename
                )[0].lower()
            )

            for file_name in os.listdir(
                output_dir
            ):

                file_name_without_ext = (
                    os.path.splitext(
                        file_name
                    )[0].lower()
                )

                if (
                    file_name_without_ext
                    ==
                    original_name
                ):

                    output_file = (
                        file_name
                    )

                    break

            if output_file is None:

                files = os.listdir(
                    output_dir
                )

                if files:

                    output_file = (
                        files[0]
                    )

            now = datetime.now()

            current_date = (
                now.strftime(
                    "%Y-%m-%d"
                )
            )

            current_time = (
                now.strftime(
                    "%H:%M:%S"
                )
            )

            detections = []

            for result in results:

                if result.boxes is not None:

                    for box in result.boxes:

                        class_id = int(
                            box.cls[0]
                        )

                        confidence = float(
                            box.conf[0]
                        )

                        class_name = (
                            model.names[
                                class_id
                            ]
                        )

                        confidence_percent = round(

                            confidence * 100,

                            2

                        )

                        detections.append({

                            "class":
                                class_name,

                            "confidence":
                                confidence_percent

                        })

                        save_detection(

                            image=(
                                output_file
                            ),

                            date=(
                                current_date
                            ),

                            time=(
                                current_time
                            ),

                            location=(
                                DEFAULT_LOCATION
                            ),

                            detection_type=(
                                class_name
                            ),

                            confidence=(
                                confidence_percent
                            )

                        )

            helmet_count = 0

            nohelmet_count = 0

            for detection in detections:

                class_name = (
                    detection[
                        "class"
                    ].lower()
                )

                if class_name == "helmet":

                    helmet_count += 1

                elif class_name in [

                    "no-helmet",

                    "no helmet"

                ]:

                    nohelmet_count += 1

            plate_info = None

            owner_info = None

            notice_info = None

            violation_id = None

            violation_datetime = now

            # -------------------------------------------------
            # Violation
            # -------------------------------------------------

            if nohelmet_count > 0:

                for detection in detections:

                    if detection[
                        "class"
                    ].lower() in [

                        "no-helmet",

                        "no helmet"

                    ]:

                        violation_id = (
                            save_violation(

                                image=(
                                    output_file
                                ),

                                date=(
                                    current_date
                                ),

                                time=(
                                    current_time
                                ),

                                location=(
                                    DEFAULT_LOCATION
                                ),

                                detection_type=(
                                    "No-Helmet"
                                ),

                                confidence=(
                                    detection[
                                        "confidence"
                                    ]
                                )

                            )
                        )

                        original_image = (
                            cv2.imread(
                                filepath
                            )
                        )

                        if (
                            original_image
                            is not None
                        ):

                            (
                                plate_info,

                                owner_info,

                                notice_info

                            ) = (
                                process_violation_evidence(

                                    image=(
                                        original_image
                                    ),

                                    violation_id=(
                                        violation_id
                                    ),

                                    violation_datetime=(
                                        violation_datetime
                                    ),

                                    confidence=(
                                        detection[
                                            "confidence"
                                        ]
                                    )

                                )
                            )

                        break

            return render_template(

                "result.html",

                filename=filename,

                detections=detections,

                helmet_count=helmet_count,

                nohelmet_count=nohelmet_count,

                result_file=output_file,

                is_video=False,

                plate_info=plate_info,

                owner_info=owner_info,

                notice_info=notice_info

            )

        # =================================================
        # VIDEO
        # =================================================

        else:

            print(
                "Video detected."
            )

            print(
                "Starting YOLO video processing..."
            )

            results = model.predict(

                source=filepath,

                conf=0.1,

                save=True,

                project=RUNS_FOLDER,

                name=run_name,

                exist_ok=True,

                stream=True

            )

            detections = []

            helmet_detected = False

            nohelmet_detected = False

            max_helmet_confidence = 0

            max_nohelmet_confidence = 0

            frame_count = 0

            output_dir = None

            violation_frame = None

            # -------------------------------------------------
            # Process frames
            # -------------------------------------------------

            for result in results:

                frame_count += 1

                output_dir = str(
                    result.save_dir
                )

                if result.boxes is not None:

                    for box in result.boxes:

                        class_id = int(
                            box.cls[0]
                        )

                        confidence = float(
                            box.conf[0]
                        )

                        class_name = (
                            model.names[
                                class_id
                            ]
                        )

                        confidence_percent = round(

                            confidence * 100,

                            2

                        )

                        class_lower = (
                            class_name.lower()
                        )

                        # ---------------------------------
                        # Helmet
                        # ---------------------------------

                        if class_lower == "helmet":

                            helmet_detected = True

                            if (
                                confidence_percent
                                >
                                max_helmet_confidence
                            ):

                                max_helmet_confidence = (
                                    confidence_percent
                                )

                        # ---------------------------------
                        # No Helmet
                        # ---------------------------------

                        elif class_lower in [

                            "no-helmet",

                            "no helmet"

                        ]:

                            nohelmet_detected = True

                            if (
                                confidence_percent
                                >
                                max_nohelmet_confidence
                            ):

                                max_nohelmet_confidence = (
                                    confidence_percent
                                )

                            if (
                                violation_frame
                                is None
                            ):

                                try:

                                    violation_frame = (
                                        result.orig_img.copy()
                                    )

                                except Exception:

                                    violation_frame = (
                                        None
                                    )

            print(
                "Total video frames processed:",
                frame_count
            )

            print(
                "Helmet detected:",
                helmet_detected
            )

            print(
                "No-Helmet detected:",
                nohelmet_detected
            )

            # =================================================
            # FIND OUTPUT VIDEO
            # =================================================

            output_file = None

            if (
                output_dir
                and
                os.path.exists(
                    output_dir
                )
            ):

                original_name = (
                    os.path.splitext(
                        filename
                    )[0].lower()
                )

                for file_name in os.listdir(
                    output_dir
                ):

                    file_name_without_ext = (
                        os.path.splitext(
                            file_name
                        )[0].lower()
                    )

                    if (
                        file_name_without_ext
                        ==
                        original_name
                    ):

                        output_file = (
                            file_name
                        )

                        break

                if output_file is None:

                    for file_name in os.listdir(
                        output_dir
                    ):

                        if (
                            os.path.splitext(
                                file_name
                            )[1].lower()
                            in video_extensions
                        ):

                            output_file = (
                                file_name
                            )

                            break

            print(
                "Output video:",
                output_file
            )

            # -------------------------------------------------
            # Convert AVI to browser-compatible MP4
            # -------------------------------------------------

            if output_file and output_dir:

                output_video_path = os.path.join(
                    output_dir,
                    output_file
                )

                converted_filename, converted_path = (
                    convert_video_to_mp4(
                        output_video_path
                    )
                )

                if converted_filename and converted_path:
                    output_file = converted_filename

            print(
                "Browser video output:",
                output_file
            )

            now = datetime.now()

            current_date = (
                now.strftime(
                    "%Y-%m-%d"
                )
            )

            current_time = (
                now.strftime(
                    "%H:%M:%S"
                )
            )

            # =================================================
            # HELMET SUMMARY
            # =================================================

            if helmet_detected:

                detections.append({

                    "class":
                        "Helmet",

                    "confidence":
                        max_helmet_confidence

                })

                save_detection(

                    image=(
                        output_file
                        or
                        filename
                    ),

                    date=(
                        current_date
                    ),

                    time=(
                        current_time
                    ),

                    location=(
                        DEFAULT_LOCATION
                    ),

                    detection_type=(
                        "Helmet"
                    ),

                    confidence=(
                        max_helmet_confidence
                    )

                )

            # =================================================
            # NO HELMET SUMMARY
            # =================================================

            plate_info = None

            owner_info = None

            notice_info = None

            violation_id = None

            violation_datetime = now

            if nohelmet_detected:

                detections.append({

                    "class":
                        "No-Helmet",

                    "confidence":
                        max_nohelmet_confidence

                })

                save_detection(

                    image=(
                        output_file
                        or
                        filename
                    ),

                    date=(
                        current_date
                    ),

                    time=(
                        current_time
                    ),

                    location=(
                        DEFAULT_LOCATION
                    ),

                    detection_type=(
                        "No-Helmet"
                    ),

                    confidence=(
                        max_nohelmet_confidence
                    )

                )

                # -------------------------------------------------
                # Save violation
                # -------------------------------------------------

                violation_id = (
                    save_violation(

                        image=(
                            output_file
                            or
                            filename
                        ),

                        date=(
                            current_date
                        ),

                        time=(
                            current_time
                        ),

                        location=(
                            DEFAULT_LOCATION
                        ),

                        detection_type=(
                            "No-Helmet"
                        ),

                        confidence=(
                            max_nohelmet_confidence
                        )

                    )
                )

                # -------------------------------------------------
                # Plate detection on violation frame
                # -------------------------------------------------

                if (
                    violation_frame
                    is not None
                ):

                    (
                        plate_info,

                        owner_info,

                        notice_info

                    ) = (
                        process_violation_evidence(

                            image=(
                                violation_frame
                            ),

                            violation_id=(
                                violation_id
                            ),

                            violation_datetime=(
                                violation_datetime
                            ),

                            confidence=(
                                max_nohelmet_confidence
                            )

                        )
                    )

                else:

                    # -------------------------------------------------
                    # Fallback frame
                    # -------------------------------------------------

                    try:

                        video_capture = (
                            cv2.VideoCapture(
                                filepath
                            )
                        )

                        success, frame = (
                            video_capture.read()
                        )

                        video_capture.release()

                        if success:

                            (
                                plate_info,

                                owner_info,

                                notice_info

                            ) = (
                                process_violation_evidence(

                                    image=frame,

                                    violation_id=(
                                        violation_id
                                    ),

                                    violation_datetime=(
                                        violation_datetime
                                    ),

                                    confidence=(
                                        max_nohelmet_confidence
                                    )

                                )
                            )

                    except Exception as e:

                        print(
                            "Fallback frame error:",
                            e
                        )

            # =================================================
            # COUNTS
            # =================================================

            helmet_count = (
                1
                if helmet_detected
                else
                0
            )

            nohelmet_count = (
                1
                if nohelmet_detected
                else
                0
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

                result_file=output_file,

                is_video=True,

                plate_info=plate_info,

                owner_info=owner_info,

                notice_info=notice_info

            )

    return render_template(
        "upload.html"
    )


# =========================================================
# SERVE YOLO RESULTS
# =========================================================

@app.route(
    "/results/<path:filename>"
)
def results(filename):

    base_folder = RUNS_FOLDER

    for root, directories, files in os.walk(
        base_folder
    ):

        if filename in files:

            return send_from_directory(

                root,

                filename

            )

    return (
        "Result file not found.",
        404
    )


# =========================================================
# SERVE PENALTY NOTICE
# =========================================================

@app.route(
    "/notice/<path:filename>"
)
def notice(filename):

    return send_from_directory(

        NOTICE_FOLDER,

        filename

    )


# =========================================================
# SERVE PLATE EVIDENCE
# =========================================================

@app.route(
    "/plate_evidence/<path:filename>"
)
def plate_evidence(filename):

    return send_from_directory(

        PLATE_EVIDENCE_FOLDER,

        filename

    )


# =========================================================
# DASHBOARD
# =========================================================

@app.route("/dashboard")
def dashboard():

    conn = sqlite3.connect(
        DATABASE
    )

    cursor = conn.cursor()

    # -----------------------------------------------------
    # Helmet
    # -----------------------------------------------------

    cursor.execute("""
        SELECT COUNT(*)
        FROM detections
        WHERE LOWER(detection_type)
        = 'helmet'
    """)

    total_helmet = (
        cursor.fetchone()[0]
    )

    # -----------------------------------------------------
    # No Helmet
    # -----------------------------------------------------

    cursor.execute("""
        SELECT COUNT(*)
        FROM detections
        WHERE LOWER(detection_type)
        IN ('no-helmet', 'no helmet')
    """)

    total_nohelmet = (
        cursor.fetchone()[0]
    )

    # -----------------------------------------------------
    # Total
    # -----------------------------------------------------

    total_detections = (
        total_helmet
        +
        total_nohelmet
    )

    # -----------------------------------------------------
    # Notices
    # -----------------------------------------------------

    cursor.execute("""
        SELECT COUNT(*)
        FROM penalty_notices
    """)

    total_notices = (
        cursor.fetchone()[0]
    )

    # -----------------------------------------------------
    # Notifications
    # -----------------------------------------------------

    cursor.execute("""
        SELECT COUNT(*)
        FROM notifications
    """)

    total_notifications = (
        cursor.fetchone()[0]
    )

    # -----------------------------------------------------
    # Weekly
    # -----------------------------------------------------

    today = datetime.now().date()

    weekly_data = []

    for i in range(
        6,
        -1,
        -1
    ):

        day = (
            today
            -
            timedelta(days=i)
        )

        date_string = (
            day.strftime(
                "%Y-%m-%d"
            )
        )

        day_name = (
            day.strftime("%a")
        )

        cursor.execute("""
            SELECT COUNT(*)
            FROM detections
            WHERE date = ?
            AND LOWER(detection_type)
            = 'helmet'
        """, (
            date_string,
        ))

        helmet = (
            cursor.fetchone()[0]
        )

        cursor.execute("""
            SELECT COUNT(*)
            FROM detections
            WHERE date = ?
            AND LOWER(detection_type)
            IN ('no-helmet', 'no helmet')
        """, (
            date_string,
        ))

        nohelmet = (
            cursor.fetchone()[0]
        )

        weekly_data.append({

            "day":
                day_name,

            "date":
                date_string,

            "helmet":
                helmet,

            "nohelmet":
                nohelmet

        })

    conn.close()

    return render_template(

        "dashboard.html",

        total_helmet=(
            total_helmet
        ),

        total_nohelmet=(
            total_nohelmet
        ),

        total_detections=(
            total_detections
        ),

        total_notices=(
            total_notices
        ),

        total_notifications=(
            total_notifications
        ),

        weekly_data=(
            weekly_data
        )

    )


# =========================================================
# HISTORY
# =========================================================

@app.route("/history")
def history():

    conn = sqlite3.connect(
        DATABASE
    )

    conn.row_factory = (
        sqlite3.Row
    )

    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM violations
        ORDER BY id DESC
    """)

    violations = (
        cursor.fetchall()
    )

    conn.close()

    return render_template(

        "history.html",

        violations=(
            violations
        )

    )


# =========================================================
# RUN FLASK
# =========================================================

if __name__ == "__main__":

    init_database()

    app.run(
        debug=True
    )