<div align="center">
  <img src="https://img.shields.io/badge/Status-Active-success.svg?style=for-the-badge" alt="Status">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-009688.svg?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Chrome-Extension-4285F4.svg?style=for-the-badge&logo=google-chrome&logoColor=white" alt="Chrome Extension">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge" alt="License">

  <br><br>

  <h1>🛡️ OpenSeek</h1>
  <p><strong>Advanced Image-Only Deepfake & AI Generation Detection Tool</strong></p>
  <br>
</div>

---

## 📖 Short Description

**OpenSeek** is a powerful, open-source forensic tool designed to help you instantly detect AI-generated images and deepfakes directly in your browser. Using a research-grade machine learning backend, it analyzes spatial inconsistencies, lighting mismatches, and AI artifacts to confidently tell you if an image is a deepfake manipulation, AI-generated (Midjourney, Stable Diffusion), or a genuine photograph.

---

## ✨ Features

- **Chrome Extension Integration**: Right-click on any image while browsing the web and select **"Analyze for Deepfake"** to instantly scan it.
- **Advanced Machine Learning**: Uses an EfficientNet-based ensemble model and Vision Transformers (ViT).
- **Face-Focused Layer**: Automatically crops and heavily scrutinizes faces for AI tampering.
- **Risk Level Scoring**: Returns a simple, easy-to-understand risk level (🟢 Low, 🟡 Medium, 🔴 High) and a percentage score.
- **Privacy First**: Images are temporarily analyzed and deleted immediately from the server; logs are kept locally in a SQLite database.

---

## 📸 Screenshots

*(Add your screenshots here!)*

> *Pro tip: Take a screenshot of the context menu and the popup showing a 99% Deepfake result, and drag-and-drop them into this section.*

---

## 🛠️ How It Works

1. **User Request**: The user right-clicks an image in the browser and clicks "Analyze for Deepfake".
2. **Data Extraction**: The Chrome Extension extracts the image data and sends it securely to the local backend.
3. **Forensic Pipeline**: The backend runs the image through face detectors, spatial frequency analyzers, and an EfficientNet ensemble model.
4. **Scoring**: A final blended confidence score is calculated.
5. **UI Rendering**: The result (Real vs. AI) is immediately passed back to the extension popup for the user to review.

---

## 💻 Tech Stack

- **Backend / Machine Learning**: Python, FastAPI, PyTorch, OpenCV, HuggingFace Transformers
- **Database**: SQLite (Local caching and request logging)
- **Frontend / Extension**: JavaScript (Manifest V3), HTML5, CSS3

---

## 🚀 Installation Guide

### Step 1: Start the Backend (FastAPI)
You need Python 3.10+ installed on your computer.

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

---

## 🎮 Usage Guide

1. Ensure the Python backend is actively running in your terminal.
2. Go to any website with images (e.g., Twitter, Google Images, Reddit).
3. **Right-click** on a suspect image and click **"Analyze for Deepfake"**.
4. The OpenSeek extension will instantly notify you if the image is real or AI-generated!

---

## 📂 Project Structure

```text
OpenSeek/
├── backend/                  # Python FastAPI Backend
│   ├── main.py               # Main API entry point
│   ├── models/               # ML Models (Image detectors)
│   ├── routers/              # API Route definitions
│   └── requirements.txt      # Python dependencies
├── extension/                # Google Chrome Extension
│   ├── background.js         # Service worker (Context Menu)
│   ├── popup.html/.js        # Extension UI
│   └── manifest.json         # Manifest V3 configuration
├── LICENSE                   # MIT License
└── README.md                 # This documentation
```

---

## 🔌 API Endpoint

If you want to integrate OpenSeek into your own app, you can use the backend directly!

**POST** `/analyze-image`
- **Payload (Form Data)**: `file` (The image file to upload)
- **Response**:
```json
{
  "status": "success",
  "result": {
    "is_deepfake": true,
    "confidence_score": 0.98,
    "risk_level": "High Risk",
    "details": {
      "face_detected": true,
      "model_version": "v1.2"
    }
  }
}
```

---

## 🗺️ Roadmap

- [ ] Add support for batch image analysis.
- [ ] Implement cloud-sync for history and analytics.
- [ ] Expand the ensemble to detect newer Flux and Midjourney v6 models.
- [ ] Port the extension to Firefox.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! 

1. Fork the Project.
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`).
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the Branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

---

## 📜 License

Distributed under the **MIT License**. See the `LICENSE` file for more information.

---

## 👥 Contributors

This project is actively maintained by [Yash Bansal](https://github.com/yashbansal-dev). Feel free to reach out for collaborations!

---

## ⭐ Star the Repo

If you find this project useful, helpful, or interesting, please consider leaving a star ⭐ on the repository! It helps the project grow and reach more people.

<div align="center">
  Built with ❤️ to keep the internet transparent.
</div>
