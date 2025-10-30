import sqlite3
import hashlib
import os

# Use project folder to store DB
DB_FILE = os.path.join(os.getcwd(), "users.db")

def init_db():
    """Initialize the database and ensure 'users' table with 'password' column exists."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # Create table if it doesn't exist
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY
        )
    """)
    # Ensure 'password' column exists
    c.execute("PRAGMA table_info(users)")
    columns = [col[1] for col in c.fetchall()]
    if "password" not in columns:
        c.execute("ALTER TABLE users ADD COLUMN password TEXT")
    conn.commit()
    conn.close()

def hash_password(password: str) -> str:
    """Hash a password with SHA256."""
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(username: str, password: str) -> bool:
    """Create a new user. Returns False if username exists."""
    init_db()  # Ensure DB and table exist
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
    """Authenticate a user. Returns True if username/password match."""
    init_db()  # Ensure DB and table exist
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username = ?", (username,))
    row = c.fetchone()
    conn.close()
    if row is None or row[0] is None:
        return False
    return row[0] == hash_password(password)
