/**
 * label-web/server.js
 * Express backend phục vụ giao diện gán nhãn ảnh bất định (low_confidence).
 *
 * API endpoints:
 *   GET  /api/images          — danh sách ảnh chờ gán nhãn (kèm metadata)
 *   GET  /api/image/:id       — serve file ảnh
 *   POST /api/label           — lưu nhãn { id, label } → data/labeled/
 *   POST /api/skip            — bỏ qua ảnh (không xóa khỏi queue)
 *   GET  /api/stats           — thống kê tiến trình
 *   GET  /health              — health check
 *
 * Sau khi gán nhãn đủ RETRAIN_THRESHOLD:
 *   → Tạo file retrain_trigger/trigger_{timestamp}.json
 *   → Gọi GitHub API dispatch event "retrain-requested" (nếu có GITHUB_TOKEN)
 */

const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const https = require('https');

const app = express();
const PORT = parseInt(process.env.PORT || '4000', 10);

// ── Cấu hình đường dẫn ────────────────────────────────────────────────────────
const LOW_CONFIDENCE_DIR = process.env.LOW_CONFIDENCE_DIR || path.join(__dirname, 'data/low_confidence');
const LABELED_DIR        = process.env.LABELED_DIR        || path.join(__dirname, 'data/labeled');
const TRIGGER_DIR        = process.env.RETRAIN_TRIGGER_DIR|| path.join(__dirname, 'data/retrain_trigger');
const RAW_DATA_DIR       = process.env.RAW_DATA_DIR       || path.join(__dirname, 'data/raw');
const RETRAIN_THRESHOLD  = parseInt(process.env.RETRAIN_THRESHOLD || '10', 10);

// GitHub dispatch config (từ environment secrets)
const GITHUB_TOKEN       = process.env.GITHUB_TOKEN || '';
const GITHUB_REPO_OWNER  = process.env.GITHUB_REPO_OWNER || '';
const GITHUB_REPO_NAME   = process.env.GITHUB_REPO_NAME  || '';

// ── Thư mục lưu trữ các đợt label đã trigger ────────────────────────────────
const LABELED_ARCHIVE_DIR = process.env.LABELED_ARCHIVE_DIR || path.join(path.dirname(LABELED_DIR), 'labeled_archive');

// ── Khởi tạo thư mục ──────────────────────────────────────────────────────────
[LOW_CONFIDENCE_DIR, LABELED_DIR, TRIGGER_DIR, LABELED_ARCHIVE_DIR,
  path.join(RAW_DATA_DIR, 'Cat'),
  path.join(RAW_DATA_DIR, 'Dog'),
].forEach(dir => {
  fs.mkdirSync(dir, { recursive: true });
});

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// ── Helper: đếm số file đã được label ────────────────────────────────────────
function countLabeled() {
  try {
    return fs.readdirSync(LABELED_DIR)
      .filter(f => f.endsWith('.json')).length;
  } catch { return 0; }
}

// ── Helper: lấy danh sách ảnh chờ gán nhãn ───────────────────────────────────
function getPendingImages() {
  try {
    const files = fs.readdirSync(LOW_CONFIDENCE_DIR);
    const ids = [...new Set(
      files
        .filter(f => /\.(png|jpg|jpeg|webp)$/i.test(f))
        .map(f => path.basename(f, path.extname(f)))
    )];

    return ids.map(id => {
      const imgFile = files.find(f =>
        path.basename(f, path.extname(f)) === id &&
        /\.(png|jpg|jpeg|webp)$/i.test(f)
      );
      const jsonFile = path.join(LOW_CONFIDENCE_DIR, `${id}.json`);

      let meta = { id, predicted_label: 'Unknown', confidence: 0, original_filename: '' };
      try {
        meta = JSON.parse(fs.readFileSync(jsonFile, 'utf8'));
      } catch { /* không có JSON thì dùng default */ }

      return {
        id,
        image_ext: path.extname(imgFile),
        predicted_label: meta.predicted_label || 'Unknown',
        confidence: meta.confidence || 0,
        original_filename: meta.original_filename || imgFile,
        timestamp: meta.timestamp || null,
      };
    }).sort((a, b) => a.confidence - b.confidence); // ưu tiên confidence thấp nhất
  } catch (err) {
    console.error('[server] Lỗi đọc low_confidence dir:', err.message);
    return [];
  }
}

// ── Helper: gọi GitHub repository_dispatch để trigger retrain ─────────────────
function triggerGitHubRetrain(labeledCount) {
  if (!GITHUB_TOKEN || !GITHUB_REPO_OWNER || !GITHUB_REPO_NAME) {
    console.log('[server] Bỏ qua GitHub dispatch: chưa cấu hình GITHUB_TOKEN/REPO_OWNER/REPO_NAME');
    return;
  }

  const payload = JSON.stringify({
    event_type: 'retrain-requested',
    client_payload: {
      labeled_count: labeledCount,
      threshold: RETRAIN_THRESHOLD,
      triggered_at: new Date().toISOString(),
    }
  });

  const options = {
    hostname: 'api.github.com',
    path: `/repos/${GITHUB_REPO_OWNER}/${GITHUB_REPO_NAME}/dispatches`,
    method: 'POST',
    headers: {
      'Authorization': `token ${GITHUB_TOKEN}`,
      'Accept': 'application/vnd.github.v3+json',
      'Content-Type': 'application/json',
      'Content-Length': Buffer.byteLength(payload),
      'User-Agent': 'label-web/1.0',
    }
  };

  const req = https.request(options, (res) => {
    console.log(`[server] GitHub dispatch → HTTP ${res.statusCode}`);
    if (res.statusCode === 204) {
      console.log('[server] ✅ Retrain job đã được trigger thành công trên GitHub Actions!');
    } else {
      console.warn(`[server] ⚠️  GitHub dispatch trả về ${res.statusCode}`);
    }
  });

  req.on('error', (err) => {
    console.error('[server] Lỗi gọi GitHub API:', err.message);
  });

  req.write(payload);
  req.end();
}

// ── Helper: archive và reset thư mục labeled/ sau khi trigger ────────────────
function resetLabeledDir(timestamp) {
  try {
    const archiveSlot = path.join(LABELED_ARCHIVE_DIR, `batch_${timestamp}`);
    fs.mkdirSync(archiveSlot, { recursive: true });

    const files = fs.readdirSync(LABELED_DIR);
    files.forEach(f => {
      fs.renameSync(
        path.join(LABELED_DIR, f),
        path.join(archiveSlot, f)
      );
    });

    console.log(`[server] 📦 Đã archive ${files.length} file label vào: ${archiveSlot}`);
    console.log('[server] 🔄 Counter reset — sẵn sàng đợt gán nhãn tiếp theo.');
  } catch (err) {
    console.error('[server] Lỗi khi reset labeled dir:', err.message);
  }
}

// ── Helper: kiểm tra và trigger retrain khi đủ ngưỡng ────────────────────────
function checkAndTriggerRetrain() {
  const labeled = countLabeled();
  if (labeled > 0 && labeled % RETRAIN_THRESHOLD === 0) {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const triggerFile = path.join(TRIGGER_DIR, `trigger_${timestamp}.json`);

    const triggerData = {
      trigger_at: new Date().toISOString(),
      labeled_count: labeled,
      threshold: RETRAIN_THRESHOLD,
      status: 'pending',
    };

    try {
      fs.writeFileSync(triggerFile, JSON.stringify(triggerData, null, 2));
      console.log(`[server] 🔥 Retrain trigger tạo tại: ${triggerFile}`);
    } catch (err) {
      console.error('[server] Lỗi tạo trigger file:', err.message);
    }

    // Gọi GitHub dispatch để trigger CI/CD retrain job
    triggerGitHubRetrain(labeled);

    // Reset counter: archive labeled/ → bắt đầu đếm lại từ 0
    resetLabeledDir(timestamp);
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// API Routes
// ═══════════════════════════════════════════════════════════════════════════════

// GET /health
app.get('/health', (req, res) => {
  res.json({ status: 'ok', service: 'label-web', timestamp: new Date().toISOString() });
});

// GET /api/stats
app.get('/api/stats', (req, res) => {
  const pending = getPendingImages();
  const labeled = countLabeled();
  const nextTriggerAt = RETRAIN_THRESHOLD - (labeled % RETRAIN_THRESHOLD || RETRAIN_THRESHOLD);

  res.json({
    pending: pending.length,
    labeled,
    retrain_threshold: RETRAIN_THRESHOLD,
    next_trigger_in: nextTriggerAt === RETRAIN_THRESHOLD ? 0 : nextTriggerAt,
  });
});

// GET /api/images — danh sách ảnh pending
app.get('/api/images', (req, res) => {
  const images = getPendingImages();
  res.json({ images, total: images.length });
});

// GET /api/image/:id — serve ảnh
app.get('/api/image/:id', (req, res) => {
  const { id } = req.params;
  const files = fs.readdirSync(LOW_CONFIDENCE_DIR);
  const imgFile = files.find(f =>
    path.basename(f, path.extname(f)) === id &&
    /\.(png|jpg|jpeg|webp)$/i.test(f)
  );

  if (!imgFile) {
    return res.status(404).json({ error: 'Image not found' });
  }

  const imgPath = path.join(LOW_CONFIDENCE_DIR, imgFile);
  const ext = path.extname(imgFile).toLowerCase();
  const mimeMap = { '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.webp': 'image/webp' };
  res.setHeader('Content-Type', mimeMap[ext] || 'image/jpeg');
  res.sendFile(path.resolve(imgPath));
});

// POST /api/label — gán nhãn
app.post('/api/label', (req, res) => {
  const { id, label } = req.body;

  if (!id || !label) {
    return res.status(400).json({ error: 'Thiếu id hoặc label' });
  }

  const allowedLabels = ['Cat', 'Dog'];
  if (!allowedLabels.includes(label)) {
    return res.status(400).json({ error: `Label phải là một trong: ${allowedLabels.join(', ')}` });
  }

  // Tìm ảnh
  let files;
  try {
    files = fs.readdirSync(LOW_CONFIDENCE_DIR);
  } catch (err) {
    return res.status(500).json({ error: 'Không đọc được thư mục low_confidence' });
  }

  const imgFile = files.find(f =>
    path.basename(f, path.extname(f)) === id &&
    /\.(png|jpg|jpeg|webp)$/i.test(f)
  );

  if (!imgFile) {
    return res.status(404).json({ error: 'Không tìm thấy ảnh' });
  }

  // Đọc metadata
  let meta = { id, predicted_label: 'Unknown', confidence: 0 };
  try {
    meta = JSON.parse(fs.readFileSync(path.join(LOW_CONFIDENCE_DIR, `${id}.json`), 'utf8'));
  } catch { /* dùng default */ }

  // Lưu kết quả gán nhãn
  const labelData = {
    id,
    image_filename: imgFile,
    human_label: label,
    predicted_label: meta.predicted_label,
    confidence: meta.confidence,
    original_filename: meta.original_filename || '',
    labeled_at: new Date().toISOString(),
  };

  try {
    fs.writeFileSync(
      path.join(LABELED_DIR, `${id}.json`),
      JSON.stringify(labelData, null, 2)
    );
  } catch (err) {
    return res.status(500).json({ error: 'Lỗi lưu kết quả label' });
  }

  // Copy ảnh vào data/raw/{label}/ để dùng trong lần retrain tiếp theo
  // ImageFolder format: data/raw/Cat/*.jpg, data/raw/Dog/*.jpg
  const destDir = path.join(RAW_DATA_DIR, label); // label là 'Cat' hoặc 'Dog'
  const destPath = path.join(destDir, imgFile);
  try {
    fs.copyFileSync(path.join(LOW_CONFIDENCE_DIR, imgFile), destPath);
    console.log(`[server] ✅ Đã copy ảnh vào dataset: ${destPath}`);
  } catch (err) {
    console.error('[server] ❌ Lỗi copy ảnh vào raw dataset:', err.message);
    // Vẫn tiếp tục — không block việc gán nhãn
  }

  // Xóa ảnh và JSON metadata khỏi low_confidence (đã được gán nhãn)
  try {
    fs.unlinkSync(path.join(LOW_CONFIDENCE_DIR, imgFile));
  } catch (err) {
    console.warn('[server] Không xóa được ảnh:', err.message);
  }
  try {
    fs.unlinkSync(path.join(LOW_CONFIDENCE_DIR, `${id}.json`));
  } catch { /* JSON có thể không tồn tại */ }

  // Kiểm tra trigger retrain
  checkAndTriggerRetrain();

  const remaining = getPendingImages().length;
  const labeled = countLabeled();

  res.json({
    success: true,
    message: `Đã gán nhãn "${label}" cho ảnh ${id}`,
    remaining,
    labeled,
    retrain_threshold: RETRAIN_THRESHOLD,
  });
});

// POST /api/skip — bỏ qua ảnh (không xóa)
app.post('/api/skip', (req, res) => {
  const { id } = req.body;
  if (!id) {
    return res.status(400).json({ error: 'Thiếu id' });
  }
  res.json({ success: true, message: `Đã bỏ qua ảnh ${id}` });
});

// ── Start server ──────────────────────────────────────────────────────────────
app.listen(PORT, '0.0.0.0', () => {
  console.log(`
╔══════════════════════════════════════════════════════╗
║           🏷️  Label Web — Active Learning UI         ║
╠══════════════════════════════════════════════════════╣
║  URL:        http://localhost:${PORT}                    ║
║  Source dir: ${LOW_CONFIDENCE_DIR}
║  Labeled to: ${LABELED_DIR}
║  Trigger at: ${RETRAIN_THRESHOLD} labeled samples                    ║
╚══════════════════════════════════════════════════════╝
  `);
});
