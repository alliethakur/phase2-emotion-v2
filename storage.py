# storage.py - SQLite session and reading logging
import sqlite3
import json
from datetime import datetime

DB_PATH = "sessions.db"

def init_db(path=DB_PATH):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            started_at TEXT,
            face_baseline TEXT,
            vocal_baseline TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            timestamp TEXT,
            face_stress REAL,
            vocal_stress REAL,
            combined_stress REAL,
            face_emotion TEXT,
            audio_emotion TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_session_start(session_id, face_baseline, vocal_baseline, path=DB_PATH):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute(
        "INSERT OR REPLACE INTO sessions (session_id, started_at, face_baseline, vocal_baseline) VALUES (?, ?, ?, ?)",
        (session_id, datetime.now().isoformat(), json.dumps(face_baseline), json.dumps(vocal_baseline))
    )
    conn.commit()
    conn.close()

_log_conn = None

def get_log_connection(path=DB_PATH):
    global _log_conn
    if _log_conn is None:
        _log_conn = sqlite3.connect(path, check_same_thread=False)
    return _log_conn

def log_reading(session_id, face_stress, vocal_stress, combined_stress, face_emotion, audio_emotion, path=DB_PATH):
    conn = get_log_connection(path)
    c = conn.cursor()
    c.execute(
        "INSERT INTO readings (session_id, timestamp, face_stress, vocal_stress, combined_stress, face_emotion, audio_emotion) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (session_id, datetime.now().isoformat(), face_stress, vocal_stress, combined_stress, face_emotion, audio_emotion)
    )
    conn.commit()