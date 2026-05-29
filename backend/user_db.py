"""
OpenSeek Dashboard Database Manager.
Handles registration, login sessions, credits, and analysis history.
"""
import sqlite3
import hashlib
import secrets
from datetime import datetime, timedelta, timezone
import json

DB_PATH = "openseek_cache.db"

def init_user_db():
    """Initialize user, session, and scans tables."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Enable foreign keys
    c.execute("PRAGMA foreign_keys = ON;")
    
    # Users table (default daily credits limit is 10)
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            credits INTEGER DEFAULT 10,
            last_reset_date TEXT DEFAULT ''
        )
    ''')
    
    # Migration: add last_reset_date column if it doesn't exist
    try:
        c.execute("ALTER TABLE users ADD COLUMN last_reset_date TEXT DEFAULT ''")
    except sqlite3.OperationalError:
        pass # already exists
        
    # Sessions table
    c.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            token TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    ''')
    
    # User Scans (History) table
    c.execute('''
        CREATE TABLE IF NOT EXISTS user_scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            filename TEXT NOT NULL,
            ai_probability REAL NOT NULL,
            risk_level TEXT NOT NULL,
            is_ai_generated INTEGER NOT NULL,
            details_json TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    ''')
    
    conn.commit()
    conn.close()

def hash_password(password: str, salt: bytes = None) -> tuple[str, str]:
    """Securely hash a password using PBKDF2 HMAC SHA-256."""
    if salt is None:
        salt = secrets.token_bytes(16)
    pwdhash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
    return pwdhash.hex(), salt.hex()

def verify_password(password: str, password_hash: str, salt_hex: str) -> bool:
    """Verify a password matches the stored hash."""
    try:
        salt = bytes.fromhex(salt_hex)
        pwdhash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
        return pwdhash.hex() == password_hash
    except Exception:
        return False

def apply_daily_credits_reset(user_id: int, conn=None):
    """Check if the user's credits should be refilled to 10 for the day."""
    should_close = False
    if conn is None:
        conn = sqlite3.connect(DB_PATH)
        should_close = True
        
    c = conn.cursor()
    c.execute("SELECT credits, last_reset_date FROM users WHERE id = ?", (user_id,))
    row = c.fetchone()
    if row:
        credits, last_reset_date = row
        current_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        if last_reset_date != current_date:
            # Refill to 10 credits and update the last reset date
            c.execute("UPDATE users SET credits = 10, last_reset_date = ? WHERE id = ?", (current_date, user_id))
            conn.commit()
            
    if should_close:
        conn.close()

def register_user(email: str, password: str) -> dict:
    """Register a new user and give them 10 starting credits."""
    email = email.strip().lower()
    if not email or not password:
        raise ValueError("Email and password are required")
        
    pwd_hash, salt_hex = hash_password(password)
    current_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute(
            "INSERT INTO users (email, password_hash, salt, credits, last_reset_date) VALUES (?, ?, ?, 10, ?)",
            (email, pwd_hash, salt_hex, current_date)
        )
        user_id = c.lastrowid
        conn.commit()
        return {"id": user_id, "email": email, "credits": 10}
    except sqlite3.IntegrityError:
        raise ValueError("Email is already registered")
    finally:
        conn.close()

def authenticate_user(email: str, password: str) -> dict:
    """Authenticate a user. Returns user info if valid, otherwise raises ValueError."""
    email = email.strip().lower()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, email, password_hash, salt, credits FROM users WHERE email = ?", (email,))
    row = c.fetchone()
    
    if not row:
        conn.close()
        raise ValueError("Invalid email or password")
        
    user_id, user_email, pwd_hash, salt_hex, credits = row
    if not verify_password(password, pwd_hash, salt_hex):
        conn.close()
        raise ValueError("Invalid email or password")
        
    # Apply daily reset check
    apply_daily_credits_reset(user_id, conn)
    
    # Re-fetch credits
    c.execute("SELECT credits FROM users WHERE id = ?", (user_id,))
    credits = c.fetchone()[0]
    
    conn.close()
    return {"id": user_id, "email": user_email, "credits": credits}

def create_session(user_id: int) -> str:
    """Create a session token valid for 30 days."""
    token = secrets.token_hex(32)
    created_at = datetime.now(timezone.utc).isoformat()
    expires_at = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO sessions (token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
        (token, user_id, created_at, expires_at)
    )
    conn.commit()
    conn.close()
    return token

def get_user_by_session(token: str) -> dict:
    """Get the user associated with a valid, non-expired session token."""
    if not token:
        return None
        
    now = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Join sessions and users to get user details for non-expired sessions
    c.execute('''
        SELECT u.id, u.email, u.credits 
        FROM sessions s 
        JOIN users u ON s.user_id = u.id 
        WHERE s.token = ? AND s.expires_at > ?
    ''', (token, now))
    row = c.fetchone()
    
    if not row:
        conn.close()
        return None
        
    user_id = row[0]
    # Apply daily reset check
    apply_daily_credits_reset(user_id, conn)
    
    # Re-fetch user details after reset to get correct credits
    c.execute("SELECT id, email, credits FROM users WHERE id = ?", (user_id,))
    row = c.fetchone()
    conn.close()
    
    return {"id": row[0], "email": row[1], "credits": row[2]}

def delete_session(token: str):
    """Delete a session (logout)."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM sessions WHERE token = ?", (token,))
    conn.commit()
    conn.close()

def check_and_deduct_credit(user_id: int, amount: int = 1, token: str = None) -> bool:
    """Check if the user has enough credits, and deduct them. Returns True if successful."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Apply daily reset check
    apply_daily_credits_reset(user_id, conn)
    
    c.execute("SELECT credits FROM users WHERE id = ?", (user_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        return False
        
    current_credits = row[0]
    if current_credits < amount:
        conn.close()
        return False
        
    new_credits = current_credits - amount
    c.execute("UPDATE users SET credits = ? WHERE id = ?", (new_credits, user_id))
    conn.commit()
    conn.close()
    return True

def add_credits(user_id: int, amount: int) -> int:
    """Add credits to a user account. Returns new balance."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Apply daily reset check
    apply_daily_credits_reset(user_id, conn)
    
    c.execute("SELECT credits FROM users WHERE id = ?", (user_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        raise ValueError("User not found")
    
    new_credits = row[0] + amount
    c.execute("UPDATE users SET credits = ? WHERE id = ?", (new_credits, user_id))
    conn.commit()
    conn.close()
    return new_credits

def log_scan(user_id: int, filename: str, ai_probability: float, risk_level: str, is_ai_generated: bool, details: dict):
    """Log an image analysis event in user history."""
    import numpy as np
    def _local_sanitize(val):
        if isinstance(val, dict):
            return {k: _local_sanitize(v) for k, v in val.items()}
        elif isinstance(val, list):
            return [_local_sanitize(v) for v in val]
        elif isinstance(val, tuple):
            return tuple(_local_sanitize(v) for v in val)
        elif isinstance(val, np.ndarray):
            return _local_sanitize(val.tolist())
        elif isinstance(val, (np.bool_, bool)):
            return bool(val)
        elif isinstance(val, (np.integer, int)):
            return int(val)
        elif isinstance(val, (np.floating, float)):
            return float(val)
        elif isinstance(val, np.generic):
            return val.item()
        return val

    timestamp = datetime.now(timezone.utc).isoformat()
    ai_probability = _local_sanitize(ai_probability)
    is_ai_generated = _local_sanitize(is_ai_generated)
    details = _local_sanitize(details)
    details_json = json.dumps(details)
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO user_scans (user_id, timestamp, filename, ai_probability, risk_level, is_ai_generated, details_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (user_id, timestamp, filename, ai_probability, risk_level, 1 if is_ai_generated else 0, details_json))
    conn.commit()
    conn.close()

def get_user_history(user_id: int, limit: int = 50) -> list[dict]:
    """Retrieve the scan history of a user, ordered by date descending."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        SELECT id, timestamp, filename, ai_probability, risk_level, is_ai_generated, details_json 
        FROM user_scans 
        WHERE user_id = ? 
        ORDER BY id DESC 
        LIMIT ?
    ''', (user_id, limit))
    rows = c.fetchall()
    conn.close()
    
    history = []
    for r in rows:
        history.append({
            "id": r[0],
            "timestamp": r[1],
            "filename": r[2],
            "ai_probability": r[3],
            "risk_level": r[4],
            "is_ai_generated": bool(r[5]),
            "details": json.loads(r[6])
        })
    return history
