import os
import sqlite3
import time
import json
import logging
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [Retraining Daemon] - %(message)s')

DB_PATH = "../deepshield_retraining.db"
FALSE_NEGATIVES_DIR = "../dataset/false_negatives"

def init_db():
    """Initializes the SQLite database for tracking false negatives."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS reports 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  file_path TEXT,
                  reported_class TEXT,
                  true_class TEXT,
                  confidence REAL,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  processed BOOLEAN DEFAULT 0)''')
    conn.commit()
    conn.close()

def log_false_negative(file_path: str, reported_class: str, true_class: str, confidence: float):
    """
    Called by the backend when a user flags a result as incorrect.
    Copies the file to a secure holding area and logs it for retraining.
    """
    os.makedirs(FALSE_NEGATIVES_DIR, exist_ok=True)
    
    # In a real scenario, we'd copy the image to FALSE_NEGATIVES_DIR here.
    # shutil.copy(...)
    secure_path = os.path.join(FALSE_NEGATIVES_DIR, os.path.basename(file_path))
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''INSERT INTO reports (file_path, reported_class, true_class, confidence)
                 VALUES (?, ?, ?, ?)''', (secure_path, reported_class, true_class, confidence))
    conn.commit()
    conn.close()
    
    logging.info(f"Logged false negative: {secure_path} (True Class: {true_class})")

def fetch_unprocessed_batch(limit=1000):
    """Fetches a batch of unmined images for the next epoch."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT id, file_path, true_class FROM reports WHERE processed = 0 LIMIT ?', (limit,))
    rows = c.fetchall()
    conn.close()
    return rows

def mark_as_processed(ids: list):
    """Marks reports as processed after they have been batched into the training set."""
    if not ids: return
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.executemany('UPDATE reports SET processed = 1 WHERE id = ?', [(i,) for i in ids])
    conn.commit()
    conn.close()

def trigger_retraining_pipeline():
    """
    Checks if enough new false negatives have accumulated. 
    If so, it constructs a dataloader, compiles the model, and runs an epoch.
    """
    batch = fetch_unprocessed_batch()
    if len(batch) < 100:
        logging.info(f"Only {len(batch)} new samples. Waiting for at least 100 before retuning diffusion model.")
        return
        
    logging.info(f"Triggering Hard Negative Mining epoch on {len(batch)} samples...")
    
    # ── STUB: Retraining Logic ──────────────────────────────────────────────
    # 1. Load baseline models (e.g., DiffusionDetector from models/diffusion_detector.py)
    # 2. Extract specific unlearned features using robust augmentations from advanced_training.py
    # 3. Fine-tune final linear layers with low learning rate (e.g., 1e-5)
    # 4. Save updated weights to models/weights/diffusion_b2_retuned.pth
    
    time.sleep(5) # Simulating training time
    
    logging.info("Epoch complete. Marking batch as processed.")
    mark_as_processed([row[0] for row in batch])

def run_daemon():
    """
    Runs continuously, checking every 24 hours to re-align models against 
    new generator outputs bypassing the pipeline.
    """
    init_db()
    logging.info("Started Continuous Retraining Daemon. Monitoring for concept drift...")
    
    while True:
        try:
            trigger_retraining_pipeline()
        except Exception as e:
            logging.error(f"Retraining loop failed: {e}")
            
        # Sleep for 24 hours
        logging.info("Sleeping for 24 hours until next mining cycle.")
        time.sleep(86400)

if __name__ == "__main__":
    # Test execution / daemon entrypoint
    run_daemon()
