import sqlite3

conn = sqlite3.connect("students.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS students (
    id INTEGER PRIMARY KEY,
    attendance REAL,
    marks REAL,
    income INTEGER,
    study_hours REAL,
    failures INTEGER,
    dropout INTEGER
)
""")
conn.commit()
