import sqlite3
import os
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATABASE = os.path.join(
    BASE_DIR,
    "helmgurad.db"
)

NOTICE_DIR = os.path.join(
    BASE_DIR,
    "notices"
)

os.makedirs(NOTICE_DIR, exist_ok=True)

# Demo violation details
plate_number = "MP04XX0569"
location = "RIT Kottayam, Kerala"

# Example violation time
violation_time = datetime.now()

# Notice generated within 2 hours
notice_time = violation_time + timedelta(minutes=30)

conn = sqlite3.connect(DATABASE)
conn.row_factory = sqlite3.Row

cursor = conn.cursor()

cursor.execute("""
SELECT *
FROM vehicle_owners
WHERE plate_number = ?
""", (plate_number,))

owner = cursor.fetchone()

conn.close()

if not owner:
    print("❌ Owner not found.")
    exit()

notice_id = "HG-" + violation_time.strftime("%Y%m%d%H%M%S")

notice_file = os.path.join(
    NOTICE_DIR,
    notice_id + ".html"
)

html = f"""
<!DOCTYPE html>
<html>
<head>

<meta charset="UTF-8">

<title>HELMGUARD Penalty Notice</title>

<style>

body {{
    font-family: Arial, sans-serif;
    background: #f4f4f4;
    padding: 40px;
}}

.notice {{
    max-width: 800px;
    margin: auto;
    background: white;
    padding: 40px;
    border: 2px solid #222;
}}

h1 {{
    text-align: center;
}}

.warning {{
    color: red;
    font-weight: bold;
}}

table {{
    width: 100%;
    border-collapse: collapse;
    margin-top: 20px;
}}

td {{
    border: 1px solid #ccc;
    padding: 12px;
}}

.footer {{
    margin-top: 30px;
    font-size: 13px;
}}

</style>

</head>

<body>

<div class="notice">

<h1>HELMGUARD</h1>

<h2>Helmet Violation Notice</h2>

<p>
<strong>Notice ID:</strong> {notice_id}
</p>

<p class="warning">
Rider detected without a helmet.
</p>

<table>

<tr>
<td><strong>Vehicle Number</strong></td>
<td>{owner["plate_number"]}</td>
</tr>

<tr>
<td><strong>Owner Name</strong></td>
<td>{owner["owner_name"]}</td>
</tr>

<tr>
<td><strong>Vehicle Type</strong></td>
<td>{owner["vehicle_type"]}</td>
</tr>

<tr>
<td><strong>Location</strong></td>
<td>{location}</td>
</tr>

<tr>
<td><strong>Violation Time</strong></td>
<td>{violation_time.strftime("%d-%m-%Y %H:%M:%S")}</td>
</tr>

<tr>
<td><strong>Notice Generated</strong></td>
<td>{notice_time.strftime("%d-%m-%Y %H:%M:%S")}</td>
</tr>

</table>

<p>
This demonstration notice records a helmet violation
detected by the HELMGUARD system.
</p>

<p>
The notice is generated within two hours of the recorded
violation time.
</p>

<div class="footer">

<strong>HELMGUARD – AI-Based Helmet Detection & Smart Traffic Monitoring System</strong>

<p>
This is a project demonstration notice and is not an
official government/RTO document.
</p>

</div>

</div>

</body>
</html>
"""

with open(
    notice_file,
    "w",
    encoding="utf-8"
) as file:

    file.write(html)

print("✅ Penalty notice generated!")
print("Notice ID:", notice_id)
print("Owner:", owner["owner_name"])
print("Vehicle:", owner["plate_number"])
print("Violation time:", violation_time)
print("Notice time:", notice_time)
print("Notice file:", notice_file)