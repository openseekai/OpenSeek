const API_BASE = window.location.origin.includes("localhost") || window.location.origin.includes("127.0.0.1")
    ? window.location.origin 
    : "https://openseek-production.up.railway.app";

class OpenSeekDashboard {
    constructor() {
        this.token = localStorage.getItem("openseek_token") || null;
        this.user = null;
        this.history = [];
        this.currentUploadFile = null;
        
        // DOM Elements
        this.authSection = document.getElementById("auth-section");
        this.dashboardSection = document.getElementById("dashboard-section");
        this.historySection = document.getElementById("history-section");
        this.headerUserInfo = document.getElementById("header-user-info");
        
        this.headerEmail = document.getElementById("header-email");
        this.headerCredits = document.getElementById("header-credits");
        this.dashboardCredits = document.getElementById("dashboard-credits");
        
        this.loginForm = document.getElementById("login-form");
        this.registerForm = document.getElementById("register-form");
        
        this.dropZone = document.getElementById("drop-zone");
        this.scanProgressBox = document.getElementById("scan-progress-box");
        this.scanProgressFill = document.getElementById("scan-progress-fill");
        this.scanStatusText = document.getElementById("scan-status-text");
        this.scanProgressPercent = document.getElementById("scan-progress-percent");
        
        this.scanResultBox = document.getElementById("scan-result-box");
        this.resultScore = document.getElementById("result-score");
        this.resultClass = document.getElementById("result-class");
        this.resultContentType = document.getElementById("result-content-type");
        this.resultFaceVerify = document.getElementById("result-face-verify");
        this.resultAnomalyScore = document.getElementById("result-anomaly-score");
        this.resultBadgeContainer = document.getElementById("result-badge-container");
        this.resultRadialGauge = document.getElementById("result-radial-gauge");
        
        this.historyTableBody = document.getElementById("history-table-body");
        this.historyEmpty = document.getElementById("history-empty");
        
        // Modal
        this.detailModal = document.getElementById("detail-modal");
        
        // Toasts
        this.toast = document.getElementById("toast-banner");
        
        this.init();
    }

    init() {
        this.setupDragAndDrop();
        
        // Theme initialization (default to light)
        this.theme = localStorage.getItem("openseek_theme") || "light";
        document.documentElement.setAttribute("data-theme", this.theme);
        this.updateThemeIcon();
        
        this.initFirebase();
        
        const downloadBtn = document.getElementById("download-extension-btn");
        if (downloadBtn) {
            downloadBtn.href = `${API_BASE}/download-extension`;
        }
        
        if (this.token) {
            this.checkSessionAndLoadDashboard();
        } else {
            this.showAuth();
        }
    }

    async initFirebase() {
        try {
            const res = await fetch(`${API_BASE}/config/firebase`);
            if (!res.ok) return;
            const config = await res.json();
            
            // Check if firebase is configured
            if (config && config.apiKey && config.projectId) {
                // Initialize Firebase Compat
                firebase.initializeApp(config);
                this.firebaseAuth = firebase.auth();
                this.googleProvider = new firebase.auth.GoogleAuthProvider();
            } else {
                console.warn("[OpenSeek] Firebase parameters not fully configured in environment. Using credentials fallback mode.");
            }
        } catch (err) {
            console.error("[OpenSeek] Failed to initialize Firebase:", err);
        }
    }

    async handleGoogleLogin() {
        // If Firebase Auth is loaded and initialized, run the Google sign-in popup flow
        if (this.firebaseAuth && this.googleProvider) {
            try {
                this.showToast("Opening Google Sign-In...");
                const result = await this.firebaseAuth.signInWithPopup(this.googleProvider);
                const user = result.user;
                const idToken = await user.getIdToken();
                
                // Send the token to the backend for verification/session creation
                const res = await fetch(`${API_BASE}/auth/firebase-login`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        id_token: idToken,
                        email: user.email || "",
                        name: user.displayName || ""
                    })
                });
                
                if (res.ok) {
                    const data = await res.json();
                    this.token = data.token;
                    this.user = data.user;
                    localStorage.setItem("openseek_token", this.token);
                    
                    // Sync to Chrome storage if chrome extension API is accessible
                    if (window.chrome && chrome.storage && chrome.storage.local) {
                        chrome.storage.local.set({ 
                            openseek_token: this.token,
                            openseek_backend_url: API_BASE
                        });
                    }
                    
                    this.showDashboard();
                    this.refreshCreditsUI(this.user.credits);
                    await this.loadHistory();
                    this.showToast(`Welcome back!`);
                } else {
                    const errData = await res.json();
                    this.showToast(errData.detail || "Google authentication failed", true);
                }
            } catch (err) {
                console.error("Firebase Auth Error:", err);
                this.showToast(err.message || "Google Sign-In failed", true);
            }
        } else {
            // Local fallback/sandbox simulation mode if credentials are not configured in Firebase console yet:
            this.showToast("Firebase credentials not configured. Opening Sandbox Google simulation...", false);
            const mockEmail = prompt("Enter a mock Google email to simulate Google Sign-in:", "googleuser@gmail.com");
            if (!mockEmail) return;
            
            try {
                const res = await fetch(`${API_BASE}/auth/firebase-login`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        id_token: "MOCK_FIREBASE_TOKEN",
                        email: mockEmail,
                        name: "Google Sandbox User"
                    })
                });
                
                if (res.ok) {
                    const data = await res.json();
                    this.token = data.token;
                    this.user = data.user;
                    localStorage.setItem("openseek_token", this.token);
                    
                    if (window.chrome && chrome.storage && chrome.storage.local) {
                        chrome.storage.local.set({ 
                            openseek_token: this.token,
                            openseek_backend_url: API_BASE
                        });
                    }
                    
                    this.showDashboard();
                    this.refreshCreditsUI(this.user.credits);
                    await this.loadHistory();
                    this.showToast(`Signed in as simulated user: ${mockEmail}`);
                } else {
                    const errData = await res.json();
                    this.showToast(errData.detail || "Google simulation failed", true);
                }
            } catch (err) {
                this.showToast("Failed to connect to backend", true);
            }
        }
    }

    toggleTheme() {
        this.theme = this.theme === "light" ? "dark" : "light";
        document.documentElement.setAttribute("data-theme", this.theme);
        localStorage.setItem("openseek_theme", this.theme);
        this.updateThemeIcon();
    }

    updateThemeIcon() {
        const icon = document.getElementById("theme-toggle-icon");
        if (!icon) return;
        if (this.theme === "dark") {
            icon.innerHTML = `<path d="M12 7c-2.76 0-5 2.24-5 5s2.24 5 5 5 5-2.24 5-5-2.24-5-5-5zM2 13h2c.55 0 1-.45 1-1s-.45-1-1-1H2c-.55 0-1 .45-1 1s.45 1 1 1zm18 0h2c.55 0 1-.45 1-1s-.45-1-1-1h-2c-.55 0-1 .45-1 1s.45 1 1 1zM11 2v2c0 .55.45 1 1 1s1-.45 1-1V2c0-.55-.45-1-1-1s-1 .45-1 1zm0 18v2c0 .55.45 1 1 1s1-.45 1-1v-2c0-.55-.45-1-1-1s-1 .45-1 1zM5.99 4.58c-.39-.39-1.03-.39-1.41 0s-.39 1.03 0 1.41l1.06 1.06c.39.39 1.03.39 1.41 0s.39-1.03 0-1.41L5.99 4.58zm12.37 12.37c-.39-.39-1.03-.39-1.41 0s-.39 1.03 0 1.41l1.06 1.06c.39.39 1.03.39 1.41 0s.39-1.03 0-1.41l-1.06-1.06zm1.06-10.96c.39-.39.39-1.03 0-1.41s-1.03-.39-1.41 0l-1.06 1.06c-.39.39-.39 1.03 0 1.41s1.03.39 1.41 0l1.06-1.06zM7.05 18.01c.39-.39.39-1.03 0-1.41s-1.03-.39-1.41 0l-1.06 1.06c-.39.39-.39 1.03 0 1.41s1.03.39 1.41 0l1.06-1.06z"/>`;
        } else {
            icon.innerHTML = `<path d="M12 3a9 9 0 1 0 9 9c0-.46-.04-.92-.1-1.36a5.389 5.389 0 0 1-4.4 2.26 5.403 5.403 0 0 1-3.14-9.8c-.44-.06-.9-.1-1.36-.1z"/>`;
        }
    }

    // Drag and Drop implementation
    setupDragAndDrop() {
        if (!this.dropZone) return;

        ['dragenter', 'dragover'].forEach(eventName => {
            this.dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                this.dropZone.classList.add('dragover');
            }, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            this.dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                this.dropZone.classList.remove('dragover');
            }, false);
        });

        this.dropZone.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files && files.length > 0) {
                this.handleFile(files[0]);
            }
        });
    }

    // Toast Notification helper
    showToast(message, isError = false) {
        if (!this.toast) return;
        this.toast.innerText = message;
        this.toast.className = `toast active ${isError ? 'toast-error' : 'toast-success'}`;
        
        setTimeout(() => {
            this.toast.classList.remove('active');
        }, 4000);
    }

    // Auth Switch
    switchAuthTab(tab) {
        const tabs = document.querySelectorAll('.auth-tab');
        tabs.forEach(t => t.classList.remove('active'));
        
        if (tab === 'login') {
            tabs[0].classList.add('active');
            this.loginForm.classList.add('active');
            this.registerForm.classList.remove('active');
        } else {
            tabs[1].classList.add('active');
            this.registerForm.classList.add('active');
            this.loginForm.classList.remove('active');
        }
    }

    // Session Management
    async checkSessionAndLoadDashboard() {
        try {
            const res = await fetch(`${API_BASE}/auth/me`, {
                headers: { 'Authorization': `Bearer ${this.token}` }
            });
            if (res.ok) {
                this.user = await res.json();
                this.showDashboard();
                this.refreshCreditsUI(this.user.credits);
                this.loadHistory();
            } else {
                // Token invalid or expired
                this.logout();
            }
        } catch (err) {
            console.error(err);
            this.showToast("Network error. Working offline.", true);
            this.showAuth();
        }
    }

    showAuth() {
        this.authSection.classList.remove('hidden');
        this.dashboardSection.classList.add('hidden');
        this.historySection.classList.add('hidden');
        this.headerUserInfo.classList.add('hidden');
    }

    showDashboard() {
        this.authSection.classList.add('hidden');
        this.dashboardSection.classList.remove('hidden');
        this.historySection.classList.remove('hidden');
        this.headerUserInfo.classList.remove('hidden');
        
        if (this.user) {
            this.headerEmail.innerText = this.user.email;
        }

        // Start polling for credit and history updates every 5 seconds
        if (!this.pollingInterval) {
            this.pollingInterval = setInterval(async () => {
                if (this.token) {
                    try {
                        const res = await fetch(`${API_BASE}/auth/me`, {
                            headers: { 'Authorization': `Bearer ${this.token}` }
                        });
                        if (res.ok) {
                            this.user = await res.json();
                            this.refreshCreditsUI(this.user.credits);
                            this.loadHistory();
                        }
                    } catch (e) {
                        console.error("Polling error:", e);
                    }
                }
            }, 5000);
        }
    }

    logout() {
        if (this.pollingInterval) {
            clearInterval(this.pollingInterval);
            this.pollingInterval = null;
        }

        fetch(`${API_BASE}/auth/logout`, {
            method: 'POST',
            headers: { 'Authorization': `Bearer ${this.token}` }
        }).catch(console.error);

        this.token = null;
        this.user = null;
        this.history = [];
        localStorage.removeItem("openseek_token");
        this.showAuth();
        this.showToast("Logged out successfully.");
    }

    // Login Form Handler
    async handleLogin(e) {
        e.preventDefault();
        const email = document.getElementById("login-email").value;
        const password = document.getElementById("login-password").value;

        try {
            const res = await fetch(`${API_BASE}/auth/login`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, password })
            });

            const data = await res.json();
            if (res.ok) {
                this.token = data.token;
                localStorage.setItem("openseek_token", this.token);
                this.user = { email: data.user.email, credits: data.user.credits };
                this.showDashboard();
                this.refreshCreditsUI(this.user.credits);
                this.loadHistory();
                this.showToast("Welcome back to OpenSeek!");
            } else {
                this.showToast(data.detail || "Authentication failed", true);
            }
        } catch (err) {
            this.showToast("Failed to connect to the authentication server.", true);
        }
    }

    // Register Form Handler
    async handleRegister(e) {
        e.preventDefault();
        const email = document.getElementById("reg-email").value;
        const password = document.getElementById("reg-password").value;

        try {
            const res = await fetch(`${API_BASE}/auth/register`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, password })
            });

            const data = await res.json();
            if (res.ok) {
                this.showToast("Registration successful! You can now log in.");
                // Auto transition to login tab & populate
                this.switchAuthTab('login');
                document.getElementById("login-email").value = email;
                document.getElementById("login-password").value = password;
            } else {
                this.showToast(data.detail || "Registration failed", true);
            }
        } catch (err) {
            this.showToast("Failed to connect to the authentication server.", true);
        }
    }

    // Update credits visual display
    refreshCreditsUI(credits) {
        if (this.user) {
            this.user.credits = credits;
        }
        
        // Update display text
        this.headerCredits.innerText = credits;
        this.dashboardCredits.innerText = credits;
        
        // Update circular conic gradient
        // Assuming maximum starts around 10 for visual ratio (10 daily credits limit)
        const maxLimit = 10;
        const percentage = Math.min(100, Math.max(0, (credits / maxLimit) * 100));
        
        const progressCircle = document.getElementById("credits-radial-progress");
        if (progressCircle) {
            progressCircle.style.background = `
                radial-gradient(closest-side, var(--bg-primary) 79%, transparent 80% 100%),
                conic-gradient(var(--accent-primary) ${percentage}%, rgba(255, 255, 255, 0.05) ${percentage}% 100%)
            `;
        }
    }



    // Load Scan History log
    async loadHistory() {
        try {
            const res = await fetch(`${API_BASE}/user/history`, {
                headers: { 'Authorization': `Bearer ${this.token}` }
            });
            if (res.ok) {
                const data = await res.json();
                this.history = data.history;
                this.renderHistory();
            }
        } catch (err) {
            console.error("Error loading history log:", err);
        }
    }

    renderHistory() {
        this.historyTableBody.innerHTML = "";
        
        if (this.history.length === 0) {
            this.historyEmpty.style.display = 'block';
            return;
        }
        
        this.historyEmpty.style.display = 'none';
        
        this.history.forEach((scan, idx) => {
            const tr = document.createElement("tr");
            
            // Format dates
            const dateStr = new Date(scan.timestamp).toLocaleString();
            
            // Score and class styling
            const authScorePercent = Math.round((1 - scan.ai_probability) * 100);
            const riskClass = this.getRiskClass(scan.risk_level);
            
            tr.innerHTML = `
                <td><strong>${this.escapeHtml(scan.filename)}</strong></td>
                <td style="font-size: 13px; color: var(--text-muted);">${dateStr}</td>
                <td>
                    <span style="font-weight: 600; color: ${scan.is_ai_generated ? 'var(--color-high)' : 'var(--color-low)'}">
                        ${scan.is_ai_generated ? 'AI Generated' : 'Authentic'}
                    </span>
                </td>
                <td style="font-weight: 500;">${authScorePercent}%</td>
                <td>
                    <span class="badge badge-${riskClass}">
                        ${scan.risk_level}
                    </span>
                </td>
                <td>
                    <button class="btn btn-secondary" style="padding: 6px 12px; font-size: 12px;" onclick="app.openDetailModal(${idx})">View Report</button>
                </td>
            `;
            this.historyTableBody.appendChild(tr);
        });
    }

    escapeHtml(text) {
        return text
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    getRiskClass(risk) {
        switch (risk.toLowerCase()) {
            case 'low': return 'low';
            case 'medium': return 'medium';
            case 'high': return 'high';
            default: return 'uncertain';
        }
    }

    // Modal popup detailed report
    openDetailModal(idx) {
        const scan = this.history[idx];
        if (!scan) return;
        
        const details = scan.details;
        const dateStr = new Date(scan.timestamp).toLocaleString();
        
        document.getElementById("modal-filename").innerText = scan.filename;
        document.getElementById("modal-timestamp").innerText = dateStr;
        
        // Gauge / Auth Score
        const authScorePercent = Math.round((1 - scan.ai_probability) * 100);
        document.getElementById("modal-authenticity").innerText = `${authScorePercent}%`;
        document.getElementById("modal-class").innerText = scan.is_ai_generated ? 'AI Generated' : 'Authentic';
        document.getElementById("modal-class").style.color = scan.is_ai_generated ? 'var(--color-high)' : 'var(--color-low)';
        
        // Risk
        const riskClass = this.getRiskClass(scan.risk_level);
        const riskBadge = document.getElementById("modal-risk-badge");
        riskBadge.innerText = scan.risk_level;
        riskBadge.className = `badge badge-${riskClass}`;
        
        // Structural Profile
        document.getElementById("modal-content-type").innerText = details.content_type || 'Photograph';
        
        // Detailed anomaly factors
        // Display values based on existing database structure or defaults
        const spatialVal = details.manipulated_regions_heatmap ? 'Spatial Anomaly Active' : '0.00%';
        document.getElementById("modal-spatial-factor").innerText = spatialVal;
        
        const ganScore = details.predicted_class === "GAN" || (scan.is_ai_generated && scan.ai_probability > 0.8) ? 'High GAN Probability' : 'Low Anomaly';
        document.getElementById("modal-gan-factor").innerText = ganScore;
        
        const noiseVal = details.embedding_anomaly_score ? `${(details.embedding_anomaly_score * 100).toFixed(2)}%` : '0.00%';
        document.getElementById("modal-lighting-factor").innerText = noiseVal;
        
        document.getElementById("modal-face-detected").innerText = details.face_detected ? 'Yes (Face Analyzed)' : 'None Detected';
        document.getElementById("modal-confidence").innerText = details.confidence_score ? `${Math.round(details.confidence_score * 100)}%` : 'N/A';
        
        this.detailModal.classList.add('active');
    }

    closeModal(e) {
        this.detailModal.classList.remove('active');
    }

    // Trigger file chooser
    handleFileSelect(e) {
        const files = e.target.files;
        if (files && files.length > 0) {
            this.handleFile(files[0]);
        }
    }

    // Handle the scanned file
    async handleFile(file) {
        if (!this.user || this.user.credits < 1) {
            this.showToast("Insufficient credits. Please top up your dashboard.", true);
            return;
        }

        this.currentUploadFile = file;
        this.scanResultBox.style.display = 'none';
        this.scanProgressBox.style.display = 'flex';
        
        // Start Progress Simulation
        let progress = 0;
        this.scanProgressFill.style.width = '0%';
        this.scanProgressPercent.innerText = '0%';
        this.scanStatusText.innerText = "Uploading image binaries...";
        
        const progressInterval = setInterval(() => {
            if (progress < 90) {
                progress += Math.floor(Math.random() * 8) + 2;
                if (progress > 90) progress = 90;
                
                this.scanProgressFill.style.width = `${progress}%`;
                this.scanProgressPercent.innerText = `${progress}%`;
                
                if (progress > 30 && progress < 60) {
                    this.scanStatusText.innerText = "Extracting spatial coordinates...";
                } else if (progress >= 60) {
                    this.scanStatusText.innerText = "Executing ensemble neural pipelines...";
                }
            }
        }, 150);

        try {
            const formData = new FormData();
            formData.append("file", file);

            const res = await fetch(`${API_BASE}/detect-image`, {
                method: 'POST',
                headers: { 'Authorization': `Bearer ${this.token}` },
                body: formData
            });

            clearInterval(progressInterval);
            
            const data = await res.json();
            if (res.ok) {
                // Set to 100%
                this.scanProgressFill.style.width = '100%';
                this.scanProgressPercent.innerText = '100%';
                this.scanStatusText.innerText = "Analysis Complete!";
                
                setTimeout(() => {
                    this.scanProgressBox.style.display = 'none';
                    this.renderScanResult(file.name, data);
                    
                    // Deduct credit local & update
                    if (data.remaining_credits !== undefined) {
                        this.refreshCreditsUI(data.remaining_credits);
                    }
                    
                    this.loadHistory();
                    this.showToast("Scan finished. Credit deducted.");
                }, 400);
                
            } else {
                this.scanProgressBox.style.display = 'none';
                this.showToast(data.detail || "Scanning failed", true);
            }
        } catch (err) {
            clearInterval(progressInterval);
            this.scanProgressBox.style.display = 'none';
            this.showToast("Failed to connect to the forensic backend.", true);
        }
    }

    // Render results in Dashboard box
    renderScanResult(filename, data) {
        this.scanResultBox.style.display = 'grid';
        
        const authScorePercent = Math.round((1 - data.ai_probability) * 100);
        this.resultScore.innerText = `${authScorePercent}%`;
        
        // Gauge circle gradient
        const riskClass = this.getRiskClass(data.risk_level);
        let color = 'var(--color-low)';
        if (riskClass === 'medium') color = 'var(--color-medium)';
        if (riskClass === 'high') color = 'var(--color-high)';
        if (riskClass === 'uncertain') color = 'var(--color-uncertain)';
        
        this.resultRadialGauge.style.background = `
            radial-gradient(closest-side, var(--bg-primary) 79%, transparent 80% 100%),
            conic-gradient(${color} ${authScorePercent}%, rgba(255, 255, 255, 0.05) ${authScorePercent}% 100%)
        `;
        
        this.resultClass.innerText = data.is_ai_generated ? 'AI Generated' : 'Authentic';
        this.resultClass.style.color = data.is_ai_generated ? 'var(--color-high)' : 'var(--color-low)';
        
        this.resultContentType.innerText = data.content_type || 'Photograph';
        this.resultFaceVerify.innerText = data.face_detected ? 'Faces Detected' : 'None Detected';
        
        const noiseVal = data.embedding_anomaly_score ? `${(data.embedding_anomaly_score * 100).toFixed(2)}%` : '0.00%';
        this.resultAnomalyScore.innerText = noiseVal;
        
        this.resultBadgeContainer.innerHTML = `
            <span class="badge badge-${riskClass}">
                ${data.risk_level}
            </span>
        `;
    }
}

// Instantiate globally
window.app = new OpenSeekDashboard();
