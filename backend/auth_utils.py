import sqlite3
import hashlib
import os

DB_FILE = "users.db"

def init_db():
    """Ensure the users table exists."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def hash_password(password: str) -> str:
    """Hash passwords using SHA-256."""
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

def create_user(username: str, password: str) -> bool:
    """Create a new user; return False if username exists."""
    init_db()
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", 
                  (username, hash_password(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def authenticate(username: str, password: str) -> bool:
    """Validate login credentials."""
    init_db()
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username = ?", (username,))
    row = c.fetchone()
    conn.close()
    if not row:
        return False
    return row[0] == hash_password(password)
