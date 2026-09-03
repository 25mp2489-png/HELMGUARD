import sqlite3
import os

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATABASE = os.path.join(
    BASE_DIR,
    "helmgurad.db"
)

plate_number = "MP04XX0569"

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

if owner:

    print("✅ Owner found!")

    print("Plate Number:", owner["plate_number"])
    print("Owner Name:", owner["owner_name"])
    print("Phone:", owner["phone"])
    print("Address:", owner["address"])
    print("Vehicle Type:", owner["vehicle_type"])

else:

    print("❌ Owner not found.")