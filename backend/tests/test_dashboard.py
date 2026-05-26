"""
OpenSeek — Integration Test Suite for User Dashboard
"""
from __future__ import annotations

import pytest
import os
import sqlite3
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

# Mock DB_PATH for tests so it doesn't modify local openseek_cache.db
TEST_DB_PATH = "test_openseek_cache.db"

@pytest.fixture(autouse=True)
def setup_test_db():
    """Setup and clean test database before/after each test."""
    import user_db
    user_db.DB_PATH = TEST_DB_PATH
    
    # Initialize clean DB structure
    user_db.init_user_db()
    
    yield
    
    # Teardown: remove test DB file
    if os.path.exists(TEST_DB_PATH):
        try:
            os.remove(TEST_DB_PATH)
        except OSError:
            pass

@pytest.fixture(scope="module")
def client():
    """Create a TestClient with mocked model pipelines."""
    fake_ensemble = MagicMock()
    fake_ensemble.forward_analyze.return_value = {
        "ai_probability": 0.20,
        "is_ai_generated": False,
        "confidence_score": 0.90,
        "content_type": "Photograph",
        "predicted_class": "Real",
        "manipulated_regions_heatmap": "",
        "patch_manipulated_count": 0,
        "embedding_anomaly_score": 0.05,
    }

    fake_face_detector = MagicMock()
    fake_face_detector.detect.return_value = []

    with (
        patch("main._ensemble", fake_ensemble, create=True),
        patch("main._face_detector", fake_face_detector, create=True),
    ):
        from main import app
        yield TestClient(app, raise_server_exceptions=False)

def test_dashboard_flow(client):
    # 1. Register a test user
    email = "tester@openseek.ai"
    password = "password123"
    
    reg_resp = client.post("/auth/register", json={"email": email, "password": password})
    assert reg_resp.status_code == 200
    assert reg_resp.json()["status"] == "success"
    
    # 2. Login to get session token
    login_resp = client.post("/auth/login", json={"email": email, "password": password})
    assert login_resp.status_code == 200
    login_data = login_resp.json()
    assert "token" in login_data
    token = login_data["token"]
    
    auth_headers = {"Authorization": f"Bearer {token}"}
    
    # 3. Get profile details (/auth/me)
    me_resp = client.get("/auth/me", headers=auth_headers)
    assert me_resp.status_code == 200
    assert me_resp.json()["email"] == email
    assert me_resp.json()["credits"] == 10

    # 4. Check history starts empty
    history_resp = client.get("/user/history", headers=auth_headers)
    assert history_resp.status_code == 200
    assert len(history_resp.json()["history"]) == 0

    # 5. Simulate adding credits (+50)
    add_resp = client.post("/user/add-credits", json={"amount": 50}, headers=auth_headers)
    assert add_resp.status_code == 200
    assert add_resp.json()["credits"] == 60

    # 6. Verify credits increase in /auth/me
    me_resp2 = client.get("/auth/me", headers=auth_headers)
    assert me_resp2.json()["credits"] == 60

    # 7. Scan an image (with headers) -> expects credit deduction and log history
    import struct
    fake_jpg = (
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        b"\xff\xd9"
    )

    # Mock compute_hash & cv2.imread for file scans to proceed without breaking on mock bytes
    with (
        patch("main.compute_hash", return_value="fake_file_hash_123"),
        patch("cv2.imread", return_value=None)
    ):
        scan_resp = client.post(
            "/detect-image",
            files={"file": ("test_real.jpg", fake_jpg, "image/jpeg")},
            headers=auth_headers
        )
        assert scan_resp.status_code == 200
        scan_data = scan_resp.json()

        # Verify credit is deducted
        assert scan_data["remaining_credits"] == 59
        
    # 8. Check that scan is added to the user's history log
    history_resp2 = client.get("/user/history", headers=auth_headers)
    assert history_resp2.status_code == 200
    history_list = history_resp2.json()["history"]
    assert len(history_list) == 1
    assert history_list[0]["filename"] == "test_real.jpg"
    assert history_list[0]["risk_level"] == "Low"
    assert history_list[0]["is_ai_generated"] is False
    
    # 9. Test unauthorized operations
    bad_headers = {"Authorization": "Bearer badtoken"}
    unauth_resp = client.get("/auth/me", headers=bad_headers)
    assert unauth_resp.status_code == 401
    
    # 10. Logout session
    logout_resp = client.post("/auth/logout", headers=auth_headers)
    assert logout_resp.status_code == 200
    
    # Verify session is deleted
    me_after_logout = client.get("/auth/me", headers=auth_headers)
    assert me_after_logout.status_code == 401
