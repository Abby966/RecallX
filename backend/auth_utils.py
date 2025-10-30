import sqlite3
import hashlib
import os

DB_FILE = os.path.join(os.getcwd(), "users.db")

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # Create table if it doesn't exist
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT
        )
    """)
    # Ensure all rows have hashed passwords (if raw passwords exist)
    c.execute("SELECT username, password FROM users")
    rows = c.fetchall()
    for username, password in rows:
        if password is None or len(password) != 64:  # not a SHA256 hash
            # reset password to a placeholder hash (force login reset)
            placeholder = hash_password("changeme123")
            c.execute("UPDATE users SET password=? WHERE username=?", (placeholder, username))
    conn.commit()
    conn.close()

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(username: str, password: str) -> bool:
    init_db()  # ensure table exists
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
    init_db()  # ensure table exists
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username = ?", (username,))
    row = c.fetchone()
    conn.close()
    if row is None:
        return False
    return row[0] == hash_password(password)
