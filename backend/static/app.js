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
        this.resultPipeline = document.getElementById("result-pipeline");
        this.resultBadgeContainer = document.getElementById("result-badge-container");
        this.resultRadialGauge = document.getElementById("result-radial-gauge");
        
        this.historyTableBody = document.getElementById("history-table-body");
        this.historyEmpty = document.getElementById("history-empty");
        
        // Modal
        this.detailModal = document.getElementById("detail-modal");
        this.extensionModal = document.getElementById("extension-modal");
        
        // Toasts
        this.toast = document.getElementById("toast-banner");
        
        this.init();
    }

    get token() {
        return this._token;
    }

    set token(val) {
        this._token = val;
        this.syncTokenToDOM();
    }

    syncTokenToDOM() {
        let syncEl = document.getElementById("openseek-sync-data");
        if (!syncEl) {
            syncEl = document.createElement("div");
            syncEl.id = "openseek-sync-data";
            syncEl.style.display = "none";
            document.body.appendChild(syncEl);
        }
        syncEl.setAttribute("data-token", this._token || "");
        syncEl.setAttribute("data-backend", API_BASE);
    }

    init() {
        this.setupDragAndDrop();
        
        // Theme initialization (forced to light theme only)
        this.theme = "light";
        document.documentElement.setAttribute("data-theme", this.theme);
        
        this.initFirebase();
        
        const downloadBtn = document.getElementById("download-extension-btn");
        if (downloadBtn) {
            downloadBtn.href = `${API_BASE}/download-extension`;
            downloadBtn.addEventListener('click', (e) => {
                this.openExtensionModal();
            });
        }
        
        if (this.token) {
            this.checkSessionAndLoadDashboard();
        } else {
            this.showAuth();
            // Ping backend immediately so it wakes up (Railway cold start)
            // while the user is still on the login page — by the time they
            // type their password, the server is already warm.
            this._wakeBackend();
        }
    }

    _wakeBackend() {
        fetch(`${API_BASE}/health`, { method: 'GET', cache: 'no-store' })
            .then(() => console.log('[OpenSeek] Backend is warm ✅'))
            .catch(() => console.warn('[OpenSeek] Backend warming up...'));
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
        // Force light mode only
    }

    updateThemeIcon() {
        // Force light mode only
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
        // Show a lightweight loading state while we verify
        this.authSection.classList.remove('hidden');
        this.dashboardSection.classList.add('hidden');

        try {
            const res = await fetch(`${API_BASE}/auth/me`, {
                headers: { 'Authorization': `Bearer ${this.token}` }
            });
            if (res.ok) {
                this.user = await res.json();
                this.showDashboard();
                this.refreshCreditsUI(this.user.credits);
                this.loadHistory();
            } else if (res.status === 401) {
                // Explicitly invalid — logout
                this.logout();
            } else {
                // Backend sleeping / error — don't log out, just show auth
                this.showToast("Backend is warming up, please try again.", true);
                this.showAuth();
            }
        } catch (err) {
            // Network error — don't log out, keep token for next attempt
            console.warn("[OpenSeek] Session check failed (network):", err.message);
            this.showToast("Could not reach server. Check your connection.", true);
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
    // Toggle Password Visibility
    togglePasswordVisibility(id, btn) {
        const input = document.getElementById(id);
        if (!input) return;
        const isPwd = input.type === 'password';
        input.type = isPwd ? 'text' : 'password';
        if (isPwd) {
            btn.classList.add('visible');
            btn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"/><line x1="1" y1="1" x2="23" y2="23"/></svg>`;
        } else {
            btn.classList.remove('visible');
            btn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>`;
        }
    }

    // Forgot Password Mock
    handleForgotPassword(e) {
        e.preventDefault();
        const email = document.getElementById("login-email").value;
        if (!email) {
            this.showToast("Please enter your email address first.", true);
            const error = document.getElementById("login-email-error");
            if (error) {
                error.innerText = "Email is required to reset password";
                error.classList.remove("hidden");
                document.getElementById("login-email").classList.add("input-error");
            }
            return;
        }
        this.showToast(`Password reset link sent to ${email} (Mock)`);
    }

    // Inline Email/Password Validations
    validateEmailFormat(email) {
        const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return re.test(email);
    }

    validateLoginEmail() {
        const input = document.getElementById("login-email");
        const error = document.getElementById("login-email-error");
        if (!input || !error) return false;

        if (!input.value.trim()) {
            error.innerText = "Email address is required";
            error.classList.remove("hidden");
            input.classList.add("input-error");
            return false;
        } else if (!this.validateEmailFormat(input.value)) {
            error.innerText = "Please enter a valid email address";
            error.classList.remove("hidden");
            input.classList.add("input-error");
            return false;
        } else {
            error.classList.add("hidden");
            input.classList.remove("input-error");
            return true;
        }
    }

    validateLoginPassword() {
        const input = document.getElementById("login-password");
        const error = document.getElementById("login-password-error");
        if (!input || !error) return false;

        if (!input.value) {
            error.innerText = "Password is required";
            error.classList.remove("hidden");
            input.parentElement.classList.add("input-error");
            return false;
        } else {
            error.classList.add("hidden");
            input.parentElement.classList.remove("input-error");
            return true;
        }
    }

    validateRegEmail() {
        const input = document.getElementById("reg-email");
        const error = document.getElementById("reg-email-error");
        if (!input || !error) return false;

        if (!input.value.trim()) {
            error.innerText = "Email address is required";
            error.classList.remove("hidden");
            input.classList.add("input-error");
            return false;
        } else if (!this.validateEmailFormat(input.value)) {
            error.innerText = "Please enter a valid email address";
            error.classList.remove("hidden");
            input.classList.add("input-error");
            return false;
        } else {
            error.classList.add("hidden");
            input.classList.remove("input-error");
            return true;
        }
    }

    validateRegPassword() {
        const input = document.getElementById("reg-password");
        const error = document.getElementById("reg-password-error");
        if (!input || !error) return false;

        if (!input.value) {
            error.innerText = "Password is required";
            error.classList.remove("hidden");
            input.parentElement.classList.add("input-error");
            return false;
        } else if (input.value.length < 6) {
            error.innerText = "Password must be at least 6 characters";
            error.classList.remove("hidden");
            input.parentElement.classList.add("input-error");
            return false;
        } else {
            error.classList.add("hidden");
            input.parentElement.classList.remove("input-error");
            return true;
        }
    }

    // Login Form Handler
    async handleLogin(e) {
        e.preventDefault();
        
        const isEmailValid = this.validateLoginEmail();
        const isPasswordValid = this.validateLoginPassword();
        if (!isEmailValid || !isPasswordValid) {
            this.showToast("Please fix the validation errors.", true);
            return;
        }

        const email = document.getElementById("login-email").value;
        const password = document.getElementById("login-password").value;

        const btn = document.getElementById("login-submit-btn");
        if (btn) {
            btn.disabled = true;
            btn.querySelector('.btn-text').classList.add('hidden');
            btn.querySelector('.btn-spinner').classList.remove('hidden');
        }

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
                this.user = { email: data.user.email, credits: data.user.credits, id: data.user.id };

                if (window.chrome && chrome.storage && chrome.storage.local) {
                    chrome.storage.local.set({ openseek_token: this.token, openseek_backend_url: API_BASE });
                }

                this.showDashboard();
                this.refreshCreditsUI(this.user.credits);
                this.showToast("Welcome back to OpenSeek!");
                this.loadHistory();
            } else {
                this.showToast(data.detail || "Authentication failed", true);
            }
        } catch (err) {
            this.showToast("Failed to connect to the authentication server.", true);
        } finally {
            if (btn) {
                btn.disabled = false;
                btn.querySelector('.btn-text').classList.remove('hidden');
                btn.querySelector('.btn-spinner').classList.add('hidden');
            }
        }
    }

    // Register Form Handler
    async handleRegister(e) {
        e.preventDefault();
        
        const isEmailValid = this.validateRegEmail();
        const isPasswordValid = this.validateRegPassword();
        if (!isEmailValid || !isPasswordValid) {
            this.showToast("Please fix the validation errors.", true);
            return;
        }

        const email = document.getElementById("reg-email").value;
        const password = document.getElementById("reg-password").value;

        const btn = document.getElementById("reg-submit-btn");
        if (btn) {
            btn.disabled = true;
            btn.querySelector('.btn-text').classList.add('hidden');
            btn.querySelector('.btn-spinner').classList.remove('hidden');
        }

        try {
            const res = await fetch(`${API_BASE}/auth/register`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, password })
            });

            const data = await res.json();
            if (res.ok) {
                this.showToast("Registration successful! You can now log in.");
                this.switchAuthTab('login');
                document.getElementById("login-email").value = email;
                document.getElementById("login-password").value = password;
                // Clear validation states
                this.validateLoginEmail();
                this.validateLoginPassword();
            } else {
                this.showToast(data.detail || "Registration failed", true);
            }
        } catch (err) {
            this.showToast("Failed to connect to the authentication server.", true);
        } finally {
            if (btn) {
                btn.disabled = false;
                btn.querySelector('.btn-text').classList.remove('hidden');
                btn.querySelector('.btn-spinner').classList.add('hidden');
            }
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
        if (!risk) return 'uncertain';
        switch (String(risk).toLowerCase()) {
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
        document.getElementById("modal-pipeline").innerText = details.pipeline || 'Ensemble Model Pipeline';
        
        this.detailModal.classList.add('active');
    }

    closeModal(e) {
        this.detailModal.classList.remove('active');
    }

    openExtensionModal() {
        if (this.extensionModal) {
            this.extensionModal.classList.add('active');
        }
    }

    closeExtensionModal(e) {
        if (this.extensionModal) {
            this.extensionModal.classList.remove('active');
        }
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
        
        if (this.resultPipeline) {
            this.resultPipeline.innerText = data.pipeline || 'Ensemble Model Pipeline';
        }
        
        this.resultBadgeContainer.innerHTML = `
            <span class="badge badge-${riskClass}">
                ${data.risk_level}
            </span>
        `;
    }
}

// Instantiate globally
window.app = new OpenSeekDashboard();
