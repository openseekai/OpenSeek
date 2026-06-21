# OpenSeek — Deployment & Verification

How to deploy the backend so real extension users get the **accurate** detector
(`haywoodsloan/ai-image-detector-deploy`, benchmarked AUC ≈ 0.94), and how to
prove the deployed accuracy.

---

## 1. Request flow (know what serves users)

```
Chrome extension  ──►  Backend (FastAPI)  ──►  inference
                       openseek-production         │
                       .up.railway.app             ├─ A) local models in the same container   (models_loaded: true)
                                                    └─ B) forwarded to COLAB_MODEL_URL          (hybrid_mode_active: true)
```

The extension calls `/detect-image` and `/analyze-image-data` and shows
`ai_probability` as the user's score. **That number is only as good as whatever
runs the model** — so pick a serving mode below and verify it.

---

## 2. Choose a serving mode

> **No GPU required.** Detection is a single image-classification forward pass
> (~0.8 s/image on CPU). GPUs only matter for training or heavy batch throughput.

### A) Lean, self-contained — RECOMMENDED (Railway-friendly)
Runs ONLY the primary detector inside the container. Same accuracy as the full
ensemble (AUC ≈ 0.94), but ~1 GB RAM and CPU-only.

- `OPENSEEK_ENGINE=lean` (the default).
- Do **not** set `COLAB_MODEL_URL`.
- The Docker build pre-caches just the detector (`download_models.py`).
- A healthy deploy reports `"models_loaded": true` at `/health`.
- Fits a small instance: **~1 GB RAM, 1 vCPU**. (Railway's smallest paid tier is
  comfortable; the 512 MB free trial is tight — bump RAM if it OOMs.)

### A-full) Heavy ensemble (optional)
Set `OPENSEEK_ENGINE=full` to also load CLIP + mediapipe + torchvision backbones
(Grad-CAM heatmaps, extra forensic signals). Needs ~3–4 GB RAM; **not** needed
for accuracy.

### B) Hybrid — external inference host (advanced)
Backend forwards each scan to a separate inference server.

- Set `COLAB_MODEL_URL=https://<your-stable-host>` on the backend (and
  `LOW_MEMORY=1` so the proxy loads no models).
- ⚠️ **Do not use an ephemeral Colab + `trycloudflare.com` tunnel for production.**
  Those die on session end (~12 h) and the URL rotates — when down, scans fall
  back to a crude FFT/ELA heuristic. Use a persistent host with a stable URL.

---

## 3. Environment variables

Copy `backend/.env.example` to `.env` and set what you need. Secrets are
git-ignored; never commit `.env` or `firebase_service_account.json`.

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | `8080` | Listen port (Docker `CMD` honours it) |
| `DEBUG` | `false` | `true` enables verbose `/health` + debug logs |
| `LOG_LEVEL` | `INFO` | `DEBUG`/`INFO`/`WARNING`/`ERROR` |
| `DB_PATH` | `openseek.db` | SQLite path; point at a mounted volume to persist (compose uses `/app/data/openseek.db`) |
| `ALLOWED_ORIGINS` | `*` | **Set explicit origins in prod**, comma-separated (your site + `chrome-extension://<id>`) |
| `RATE_LIMIT` | `30/minute` | Per-IP limit on scan endpoints |
| `MAX_IMAGE_SIZE_MB` | `10` | Upload/download size cap (413 over it) |
| `OPENSEEK_ENGINE` | `lean` | `lean` = one detector (CPU, ~1 GB); `full` = heavy ensemble (~3–4 GB) |
| `OPENSEEK_DETECTOR_MODEL` | `haywoodsloan/ai-image-detector-deploy` | Primary detector; override to swap models |
| `COLAB_MODEL_URL` | _(unset)_ | If set, forward inference to this URL (mode B) |
| `LOW_MEMORY` | _(unset)_ | `1` skips loading models (proxy-only instances) |
| `SENTRY_DSN` | _(unset)_ | Enables error tracking when set (else no-op) |
| `FIREBASE_SERVICE_ACCOUNT_JSON` | _(unset)_ | Inline service-account JSON (or provide the file). Unset ⇒ local SQLite users |
| `FIREBASE_API_KEY` … `FIREBASE_APP_ID` | _(unset)_ | Firebase web config for Google sign-in |

---

## 4. Deploy (Docker / Cloud Run / Railway)

```bash
# Build (pre-downloads model weights into the image)
docker build -t openseek-backend .

# Run
docker run -p 8080:8080 \
  -e ALLOWED_ORIGINS="https://your-frontend,chrome-extension://<id>" \
  -e DB_PATH=/app/data/openseek.db \
  -v openseek-data:/app/data \
  openseek-backend
```

Or `cd backend && docker compose up --build` (compose includes a volume,
healthcheck, and resource limits).

After deploy, sanity-check:

```bash
curl -s https://<your-host>/health
# want: {"status":"ok","models_loaded":true, ...}        (mode A)
#   or: {"status":"ok","hybrid_mode_active":true, ...}    (mode B, with a LIVE url)
```

---

## 5. Verify accuracy on the LIVE server

Numbers must come from the deployed endpoint, not just local code. Build a small
labelled set (`dataset/real/*.jpg` genuine photos, `dataset/fake/*.jpg` AI
images — include AI **non-faces**/diffusion images, not only faces), then:

```bash
cd backend
python scripts/benchmark.py \
  --live-url https://<your-host> \
  --data_dir path/to/dataset \
  --email you@example.com --password '••••'     # or --token <session-token>
```

Reports accuracy / precision / recall / F1 / ROC-AUC for what users actually get.
Also confirm the label mapping on any model with:

```bash
python scripts/benchmark.py --verify-labels some_image.jpg
```

---

## 6. Pre-launch checklist

- [ ] `/health` shows `models_loaded: true` **or** a stable `colab_url` (not a `trycloudflare.com` tunnel)
- [ ] `ALLOWED_ORIGINS` set to explicit origins (not `*`)
- [ ] `--live-url` benchmark run on a real labelled set; AUC/accuracy recorded
- [ ] Secrets via env only (`.env` / `firebase_service_account.json` not committed)
- [ ] CI green (`.github/workflows/ci.yml`: ruff + pytest)
- [ ] Honest accuracy claim that matches the measured live number

> Accuracy is only "high" once it's **measured on a real, diverse, labelled set
> against the live endpoint.** The repo's old `data_smoke/` is random noise — do
> not benchmark against it.
