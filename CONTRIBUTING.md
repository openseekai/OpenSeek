# Contributing to OpenSeek

First off, thank you for considering contributing to OpenSeek! It's people like you who make OpenSeek a powerful tool for the community to detect deepfakes and AI-generated media.

Please take a moment to review this document in order to make the contribution process easy and effective for everyone.

---

## 🛠️ Code of Conduct
By participating in this project, you agree to abide by our code of conduct, which promotes a welcoming, friendly, and harassment-free environment for all contributors.

---

## 🚀 How Can I Contribute?

### 1. Reporting Bugs
* Search the existing issues to see if the bug has already been reported.
* If not, open a new issue with a descriptive title.
* Include clear steps to reproduce the bug, the expected behavior, and screenshots or console logs if applicable.

### 2. Suggesting Enhancements
* Open a feature request issue explaining the feature and why it would be valuable.
* Discuss with the maintainers before writing code to ensure it fits the project's vision.

### 3. Submitting Code Changes (Pull Requests)
We follow a standard GitHub Fork & Pull Request workflow:

#### A. Set Up Local Environment
1. **Fork** the repository on GitHub.
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/OpenSeekAI.git
   cd OpenSeekAI
   ```
3. **Backend Setup**:
   ```bash
   cd backend
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   python download_models.py # Downloads HF & local weights
   ```
4. **Extension Setup**:
   * Open Google Chrome.
   * Go to `chrome://extensions`.
   * Enable **Developer mode** (top-right toggle).
   * Click **Load unpacked** and select the `extension` folder of this repository.

#### B. Development Workflow
1. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   # OR
   git checkout -b fix/your-bugfix-name
   ```
2. Write clean code and ensure existing tests pass:
   ```bash
   backend/venv/bin/pytest backend/tests/
   ```
3. Commit your changes with descriptive messages:
   ```bash
   git commit -m "feat: add support for local image analysis caching"
   ```
4. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
5. Open a **Pull Request (PR)** against our `main` branch.

---

## 🎨 Design and Coding Style
* **Backend (Python)**: Follow PEP 8 guidelines. Use type hints where appropriate.
* **Frontend (Extension)**: Keep HTML/CSS/JS modular. Utilize the premium custom variable design system defined in `popup.css`.
* **Testing**: Write pytest tests for any new backend endpoint features.

Thank you for helping make OpenSeek better for everyone! 🛡️
