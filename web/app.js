/**
 * web/app.js
 * Logic frontend: kiểm tra trạng thái API, xử lý upload ảnh, gọi /predict, hiển thị kết quả.
 */

const API_BASE = 'http://localhost:8000';

// ─── DOM Elements ─────────────────────────────────────────────────────────────
const statusBadge = document.getElementById('status-badge');
const statusLabel = document.getElementById('status-label');
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const uploadHint = document.getElementById('upload-hint');
const previewCont = document.getElementById('preview-container');
const previewImg = document.getElementById('preview-img');
const clearBtn = document.getElementById('clear-btn');
const predictBtn = document.getElementById('predict-btn');
const btnText = document.getElementById('btn-text');
const btnSpinner = document.getElementById('btn-spinner');
const resultCard = document.getElementById('result-card');
const resultEmoji = document.getElementById('result-emoji');
const resultLabel = document.getElementById('result-label');
const uncertainBadge = document.getElementById('uncertain-badge');
const confidenceBar = document.getElementById('confidence-bar');
const confidenceVal = document.getElementById('confidence-value');
const savedBadge = document.getElementById('saved-badge');
const certainBadge = document.getElementById('certain-badge');
const retryBtn = document.getElementById('retry-btn');
const errorCard = document.getElementById('error-card');
const errorMsg = document.getElementById('error-msg');
const errorRetryBtn = document.getElementById('error-retry-btn');

let selectedFile = null;

// ─── Health Check ──────────────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(3000) });
    if (res.ok) {
      statusBadge.className = 'status-badge status-online';
      statusLabel.textContent = 'API Online';
    } else {
      setOffline();
    }
  } catch {
    setOffline();
  }
}

function setOffline() {
  statusBadge.className = 'status-badge status-offline';
  statusLabel.textContent = 'API Offline';
}

// Kiểm tra mỗi 5 giây
checkHealth();
setInterval(checkHealth, 5000);

// ─── File Handling ─────────────────────────────────────────────────────────────
function showPreview(file) {
  selectedFile = file;
  const url = URL.createObjectURL(file);
  previewImg.src = url;
  uploadHint.classList.add('hidden');
  previewCont.classList.remove('hidden');
  predictBtn.disabled = false;
  btnText.textContent = 'Phân loại ngay ✨';
  resultCard.classList.add('hidden');
  errorCard.classList.add('hidden');
}

function clearPreview() {
  selectedFile = null;
  previewImg.src = '';
  uploadHint.classList.remove('hidden');
  previewCont.classList.add('hidden');
  predictBtn.disabled = true;
  btnText.textContent = 'Chọn ảnh để phân loại';
  resultCard.classList.add('hidden');
  errorCard.classList.add('hidden');
  fileInput.value = '';
}

// Click on label opens file picker (label wraps input)
fileInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (file) showPreview(file);
});

clearBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  clearPreview();
});

// ─── Drag & Drop ───────────────────────────────────────────────────────────────
dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});

['dragleave', 'dragend'].forEach((ev) =>
  dropZone.addEventListener(ev, () => dropZone.classList.remove('drag-over'))
);

dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) {
    showPreview(file);
  }
});

// ─── Predict ───────────────────────────────────────────────────────────────────
predictBtn.addEventListener('click', predict);

async function predict() {
  if (!selectedFile) return;

  // Loading state
  btnText.textContent = 'Đang phân tích...';
  btnSpinner.classList.remove('hidden');
  predictBtn.disabled = true;
  resultCard.classList.add('hidden');
  errorCard.classList.add('hidden');

  try {
    const formData = new FormData();
    formData.append('file', selectedFile);

    const res = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Lỗi không xác định' }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    showResult(data);

  } catch (err) {
    showError(err.message);
  } finally {
    btnText.textContent = 'Phân loại ngay ✨';
    btnSpinner.classList.add('hidden');
    predictBtn.disabled = false;
  }
}

// ─── Show Result ───────────────────────────────────────────────────────────────
function showResult(data) {
  const isCat = data.label === 'Cat';
  resultEmoji.textContent = isCat ? '🐱' : '🐶';
  resultLabel.textContent = isCat ? 'Mèo (Cat)' : 'Chó (Dog)';
  resultLabel.className = `result-label ${isCat ? 'cat' : 'dog'}`;

  const pct = Math.round(data.confidence * 100);
  confidenceVal.textContent = `${pct}%`;

  // Animate bar after render
  requestAnimationFrame(() => {
    confidenceBar.style.width = `${pct}%`;

    // Color cue based on confidence
    if (pct >= 85) {
      confidenceBar.style.background = 'linear-gradient(90deg, #8b5cf6, #06b6d4)';
    } else if (pct >= 60) {
      confidenceBar.style.background = 'linear-gradient(90deg, #f59e0b, #fb923c)';
    } else {
      confidenceBar.style.background = 'linear-gradient(90deg, #ef4444, #f97316)';
    }
  });

  // Uncertain badge
  if (data.uncertain) {
    uncertainBadge.classList.remove('hidden');
    certainBadge.classList.add('hidden');
  } else {
    uncertainBadge.classList.add('hidden');
    certainBadge.classList.remove('hidden');
  }

  // Saved for review badge
  if (data.saved_for_review) {
    savedBadge.classList.remove('hidden');
  } else {
    savedBadge.classList.add('hidden');
  }

  resultCard.classList.remove('hidden');
}

// ─── Show Error ────────────────────────────────────────────────────────────────
function showError(msg) {
  errorMsg.textContent = msg;
  errorCard.classList.remove('hidden');
}

// ─── Retry Buttons ─────────────────────────────────────────────────────────────
retryBtn.addEventListener('click', clearPreview);
errorRetryBtn.addEventListener('click', () => {
  errorCard.classList.add('hidden');
});
