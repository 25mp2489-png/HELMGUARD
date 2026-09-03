import sqlite3
import os

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATABASE = os.path.join(
    BASE_DIR,
    "helmgurad.db"
)

conn = sqlite3.connect(DATABASE)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS vehicle_owners (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plate_number TEXT UNIQUE NOT NULL,
    owner_name TEXT NOT NULL,
    phone TEXT,
    address TEXT,
    vehicle_type TEXT
)
""")

owners = [
    (
        "MP04XX0569",
        "Demo Vehicle Owner",
        "9876543210",
        "Demo Address, Kerala",
        "Motorcycle"
    )
]

for owner in owners:
    cursor.execute("""
    INSERT OR IGNORE INTO vehicle_owners
    (plate_number, owner_name, phone, address, vehicle_type)
    VALUES (?, ?, ?, ?, ?)
    """, owner)

conn.commit()
conn.close()

print("Vehicle owner database created successfully!")
print("Database:", DATABASE)
print("Demo owner added.")