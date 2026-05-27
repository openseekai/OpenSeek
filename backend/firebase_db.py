"""
OpenSeek — Firebase Firestore User Database
Replaces the SQLite user_db.py for persistent auth across server restarts.

Performance optimizations:
  - In-memory session cache (60s TTL) — /auth/me becomes ~1ms after first hit
  - Reduced Firestore round trips in authenticate_user and get_user_by_session
  - apply_daily_credits_reset only writes if date actually changed (lazy)
  - create_session uses already-known user data (no extra Firestore read)

Collections used in Firestore:
  users/    — {email, password_hash, salt, credits, last_reset_date, created_at}
  sessions/ — {token, user_id, user_email, created_at, expires_at}
  scans/    — {user_id, timestamp, filename, ai_probability, risk_level, is_ai_generated, details}
"""

import hashlib
import secrets
import json
import os
import time
from datetime import datetime, timedelta, timezone
from threading import Lock

# ── Firebase Admin SDK ────────────────────────────────────────────────────────

import firebase_admin
from firebase_admin import credentials, firestore

_firestore_client = None


def _get_db():
    """Lazy-initialize and return the Firestore client."""
    global _firestore_client
    if _firestore_client is not None:
        return _firestore_client

    if not firebase_admin._apps:
        sa_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
        if sa_json:
            try:
                sa_dict = json.loads(sa_json)
                cred = credentials.Certificate(sa_dict)
                firebase_admin.initialize_app(cred)
                print("[OpenSeek DB] ✅ Firebase initialized from FIREBASE_SERVICE_ACCOUNT_JSON env var.")
            except Exception as e:
                raise RuntimeError(f"Failed to initialize Firebase from env JSON: {e}")
        else:
            sa_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "firebase_service_account.json")
            if os.path.exists(sa_path):
                cred = credentials.Certificate(sa_path)
                firebase_admin.initialize_app(cred)
                print(f"[OpenSeek DB] ✅ Firebase initialized from file: {sa_path}")
            else:
                firebase_admin.initialize_app()
                print("[OpenSeek DB] ✅ Firebase initialized with Application Default Credentials.")

    _firestore_client = firestore.client()
    return _firestore_client


# ── In-Memory Session Cache ───────────────────────────────────────────────────
# Avoids hitting Firestore on every /auth/me poll (every 5s from frontend).
# Cache entry: { user_id, email, credits, expires_at, cached_at }

_SESSION_CACHE: dict[str, dict] = {}   # token → cached user dict
_SESSION_CACHE_TTL = 60                 # seconds before re-validating with Firestore
_session_cache_lock = Lock()


def _cache_set(token: str, user: dict) -> None:
    with _session_cache_lock:
        _SESSION_CACHE[token] = {**user, "_cached_at": time.monotonic()}


def _cache_get(token: str) -> dict | None:
    with _session_cache_lock:
        entry = _SESSION_CACHE.get(token)
        if entry and (time.monotonic() - entry["_cached_at"]) < _SESSION_CACHE_TTL:
            return {k: v for k, v in entry.items() if k != "_cached_at"}
        if entry:
            del _SESSION_CACHE[token]  # stale
        return None


def _cache_invalidate(token: str) -> None:
    with _session_cache_lock:
        _SESSION_CACHE.pop(token, None)


def _cache_update_credits(token: str, new_credits: int) -> None:
    """Update credits in cache after a deduction without invalidating."""
    with _session_cache_lock:
        if token in _SESSION_CACHE:
            _SESSION_CACHE[token]["credits"] = new_credits


# ── Password Hashing ──────────────────────────────────────────────────────────

def hash_password(password: str, salt: bytes = None) -> tuple[str, str]:
    if salt is None:
        salt = secrets.token_bytes(16)
    pwdhash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
    return pwdhash.hex(), salt.hex()


def verify_password(password: str, password_hash: str, salt_hex: str) -> bool:
    try:
        salt = bytes.fromhex(salt_hex)
        pwdhash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
        return pwdhash.hex() == password_hash
    except Exception:
        return False


# ── Daily Credit Reset ────────────────────────────────────────────────────────

def _maybe_reset_credits(user_ref, data: dict) -> dict:
    """
    Check if daily reset is needed. If yes, write to Firestore and return
    updated data dict. If no, return the same data dict unchanged.
    This avoids the extra Firestore read that the old apply_daily_credits_reset did.
    """
    current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if data.get("last_reset_date", "") != current_date:
        user_ref.update({"credits": 10, "last_reset_date": current_date})
        return {**data, "credits": 10, "last_reset_date": current_date}
    return data


# ── Registration ──────────────────────────────────────────────────────────────

def register_user(email: str, password: str) -> dict:
    """Register a new user. Raises ValueError if email already taken."""
    email = email.strip().lower()
    if not email or not password:
        raise ValueError("Email and password are required")

    db = _get_db()
    users_ref = db.collection("users")

    existing = users_ref.where("email", "==", email).limit(1).get()
    if list(existing):
        raise ValueError("Email is already registered")

    pwd_hash, salt_hex = hash_password(password)
    current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    now_iso = datetime.now(timezone.utc).isoformat()

    doc_ref = users_ref.document()
    doc_ref.set({
        "email": email,
        "password_hash": pwd_hash,
        "salt": salt_hex,
        "credits": 10,
        "last_reset_date": current_date,
        "created_at": now_iso,
    })

    return {"id": doc_ref.id, "email": email, "credits": 10}


# ── Authentication ────────────────────────────────────────────────────────────

def authenticate_user(email: str, password: str) -> dict:
    """
    Verify credentials and return user info. Raises ValueError on failure.
    Optimized: single Firestore read for user lookup + inline reset (no extra read).
    """
    email = email.strip().lower()
    db = _get_db()

    docs = list(db.collection("users").where("email", "==", email).limit(1).get())
    if not docs:
        raise ValueError("Invalid email or password")

    doc = docs[0]
    data = doc.to_dict()

    if not verify_password(password, data["password_hash"], data["salt"]):
        raise ValueError("Invalid email or password")

    # Apply daily reset inline — no extra Firestore read needed
    data = _maybe_reset_credits(doc.reference, data)

    return {"id": doc.id, "email": data["email"], "credits": data["credits"]}


# ── Sessions ──────────────────────────────────────────────────────────────────

def create_session(user_id: str, user_email: str = "", user_credits: int = 10) -> str:
    """
    Create a 30-day session token stored in Firestore.
    Accepts optional pre-known user_email/credits to skip an extra Firestore read.
    """
    token = secrets.token_hex(32)
    now = datetime.now(timezone.utc)
    db = _get_db()

    # Only fetch user doc if email wasn't passed in
    if not user_email:
        user_doc = db.collection("users").document(user_id).get()
        if user_doc.exists:
            d = user_doc.to_dict()
            user_email = d.get("email", "")
            user_credits = d.get("credits", 10)

    db.collection("sessions").document(token).set({
        "token": token,
        "user_id": user_id,
        "user_email": user_email,
        "created_at": now.isoformat(),
        "expires_at": (now + timedelta(days=30)).isoformat(),
    })

    # Pre-warm the session cache so first /auth/me is instant
    _cache_set(token, {"id": user_id, "email": user_email, "credits": user_credits})

    return token


def get_user_by_session(token: str) -> dict | None:
    """
    Validate a session token and return user info, or None if invalid/expired.
    Uses in-memory cache (60s TTL) to avoid hitting Firestore on every poll.
    """
    if not token:
        return None

    # ── Cache hit (fast path) ──────────────────────────────────────────────
    cached = _cache_get(token)
    if cached:
        return cached

    # ── Cache miss — go to Firestore ──────────────────────────────────────
    db = _get_db()
    session_doc = db.collection("sessions").document(token).get()
    if not session_doc.exists:
        return None

    session = session_doc.to_dict()
    now_iso = datetime.now(timezone.utc).isoformat()
    if session.get("expires_at", "") < now_iso:
        session_doc.reference.delete()
        return None

    user_id = session["user_id"]
    user_ref = db.collection("users").document(user_id)
    user_doc = user_ref.get()
    if not user_doc.exists:
        return None

    data = user_doc.to_dict()
    # Apply daily reset inline — no second Firestore read
    data = _maybe_reset_credits(user_ref, data)

    result = {"id": user_doc.id, "email": data["email"], "credits": data["credits"]}
    _cache_set(token, result)
    return result


def delete_session(token: str) -> None:
    """Delete a session (logout)."""
    _cache_invalidate(token)
    db = _get_db()
    db.collection("sessions").document(token).delete()


# ── Credits ───────────────────────────────────────────────────────────────────

def check_and_deduct_credit(user_id: str, amount: int = 1, token: str = None) -> bool:
    """
    Deduct credits atomically. Returns False if insufficient.
    Pass token to also update the session cache so the UI reflects it immediately.
    """
    db = _get_db()
    user_ref = db.collection("users").document(user_id)

    @firestore.transactional
    def _deduct(transaction, ref):
        snapshot = ref.get(transaction=transaction)
        if not snapshot.exists:
            return False, 0
        credits = snapshot.to_dict().get("credits", 0)
        if credits < amount:
            return False, credits
        new_credits = credits - amount
        transaction.update(ref, {"credits": new_credits})
        return True, new_credits

    transaction = db.transaction()
    success, new_credits = _deduct(transaction, user_ref)

    # Keep cache in sync so the next /auth/me poll returns the right value
    if success and token:
        _cache_update_credits(token, new_credits)

    return success


def add_credits(user_id: str, amount: int) -> int:
    """Add credits to a user. Returns new balance."""
    db = _get_db()
    user_ref = db.collection("users").document(user_id)

    @firestore.transactional
    def _add(transaction, ref):
        snapshot = ref.get(transaction=transaction)
        if not snapshot.exists:
            raise ValueError("User not found")
        new_credits = snapshot.to_dict().get("credits", 0) + amount
        transaction.update(ref, {"credits": new_credits})
        return new_credits

    transaction = db.transaction()
    return _add(transaction, user_ref)


# ── Scan History ──────────────────────────────────────────────────────────────

def log_scan(user_id: str, filename: str, ai_probability: float,
             risk_level: str, is_ai_generated: bool, details: dict) -> None:
    """Log a scan event in the user's history collection."""
    db = _get_db()
    db.collection("scans").add({
        "user_id": user_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "filename": filename,
        "ai_probability": ai_probability,
        "risk_level": risk_level,
        "is_ai_generated": is_ai_generated,
        "details": details,
    })


def get_user_history(user_id: str, limit: int = 50) -> list[dict]:
    """Retrieve the most recent scan history for a user."""
    db = _get_db()
    docs = (
        db.collection("scans")
        .where("user_id", "==", user_id)
        .get()
    )
    history = []
    for doc in docs:
        d = doc.to_dict()
        history.append({
            "id": doc.id,
            "timestamp": d.get("timestamp", ""),
            "filename": d.get("filename", ""),
            "ai_probability": d.get("ai_probability", 0.0),
            "risk_level": d.get("risk_level", "Low"),
            "is_ai_generated": d.get("is_ai_generated", False),
            "details": d.get("details", {}),
        })
    # Sort by timestamp descending and apply limit in memory
    history.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return history[:limit]


# ── Google/Firebase Auth Login ────────────────────────────────────────────────

def get_or_create_firebase_user(email: str) -> dict:
    """
    Find an existing user by email or auto-create one for Firebase/Google sign-in.
    Optimized: inline daily reset, no extra Firestore read.
    """
    email = email.strip().lower()
    db = _get_db()

    docs = list(db.collection("users").where("email", "==", email).limit(1).get())
    if docs:
        doc = docs[0]
        data = doc.to_dict()
        data = _maybe_reset_credits(doc.reference, data)
        return {"id": doc.id, "email": data["email"], "credits": data["credits"]}

    random_pwd = secrets.token_hex(16)
    return register_user(email, random_pwd)


# ── Compatibility shim ────────────────────────────────────────────────────────

def init_user_db():
    """No-op for Firestore — eagerly connects to catch config errors on startup."""
    try:
        _get_db()
        print("[OpenSeek DB] ✅ Firestore connection established.")
    except Exception as e:
        print(f"[OpenSeek DB] ⚠️  Firestore init failed (will fall back to SQLite): {e}")
        raise
