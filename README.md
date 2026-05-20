<div align="center">
  <h1>🛡️ OpenSeek (OpenSeek)</h1>
  <p><strong>Advanced Image-Only Deepfake & AI Generation Detection Tool</strong></p>
  <br>
</div>

Welcome to **OpenSeek** (internally known as OpenSeek)! This is a powerful, open-source tool designed to help you instantly detect AI-generated images and deepfakes directly in your browser. 

Whether an image is a deepfake manipulation, created by diffusion models (like Midjourney or Stable Diffusion), or a genuine photograph, OpenSeek uses a research-grade machine learning backend to analyze the image and tell you the truth.

---

## 🚀 Features

- **Chrome Extension Integration**: Right-click on any image while browsing the web and select **"Analyze for Deepfake"** to instantly scan it.
- **Advanced Machine Learning**: Uses an EfficientNet-based ensemble model to analyze spatial inconsistencies, lighting mismatches, and AI artifacts.
- **Face-Focused Layer**: Automatically crops and heavily scrutinizes faces for AI tampering.
- **Risk Level Scoring**: Returns a simple, easy-to-understand risk level (🟢 Low, 🟡 Medium, 🔴 High) and a percentage score.
- **Privacy First**: Images are temporarily analyzed and deleted immediately from the server; logs are kept locally in a SQLite database.

---

## 🛠️ How It Works

The project is split into two parts:
1. **The Backend (`/backend`)**: A fast, local API server built with Python and FastAPI. It runs the PyTorch machine learning models.
2. **The Extension (`/extension`)**: A Chrome extension that allows you to trigger scans from any webpage without leaving your browser.

---

## 📖 Installation Guide

### Step 1: Start the Backend (FastAPI)
You need Python installed on your computer.

1. Open your terminal and navigate to the backend folder:
   ```bash
   cd backend
   ```
2. Create and activate a virtual environment:
   ```bash
   # On Mac/Linux:
   python3 -m venv venv
   source venv/bin/activate
   
   # On Windows:
   python -m venv venv
   venv\Scripts\activate
   ```
3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the server!
   ```bash
   uvicorn main:app --reload --port 8000
   ```
   *Your backend is now running at `http://localhost:8000`.*

### Step 2: Install the Chrome Extension
1. Open Google Chrome and go to `chrome://extensions/` in your address bar.
2. Turn on **Developer mode** using the toggle switch in the top right corner.
3. Click the **Load unpacked** button in the top left.
4. Select the `extension` folder from this repository.

### Step 3: Start Scanning!
1. Keep the backend terminal running.
2. Go to any website with images (like Google Images, Twitter, etc.).
3. Right-click on an image and click **"Analyze for Deepfake"**.
4. A notification and a small overlay will appear telling you if the image is real or AI-generated!

---

## 🤝 Contributing

This project is fully open-source under the **MIT License**. Contributions, bug reports, and pull requests are highly welcome! 

1. Fork the project.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

---
*Built with ❤️ to keep the internet transparent.*
