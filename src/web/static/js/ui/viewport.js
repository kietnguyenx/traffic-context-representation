/**
 * Component Sân Khấu Video 16:9 & Thanh Tiến Độ Mảnh
 * Quản lý chuyển đổi mượt mà giữa 4 chế độ: Standby -> Poster Preview -> Live Binary Stream -> H.264 Playback
 */

const standbyEl = document.getElementById("viewport-standby");
const standbyIcon = document.getElementById("standby-icon");
const standbyTitle = document.getElementById("standby-title");
const standbyDesc = document.getElementById("standby-desc");
const liveImg = document.getElementById("live-stream-frame");
const videoPlayer = document.getElementById("analyzed-video-player");

const progressBar = document.getElementById("slim-progress-bar");
const progressLabel = document.getElementById("progress-label");
const progressFill = document.getElementById("progress-fill");
const progressMeta = document.getElementById("progress-meta");

let currentBlobUrl = null;

function revokeBlobUrl() {
  if (currentBlobUrl) {
    URL.revokeObjectURL(currentBlobUrl);
    currentBlobUrl = null;
  }
}

export function showStandbyViewport(isWarning = false) {
  revokeBlobUrl();
  videoPlayer.pause();
  videoPlayer.removeAttribute("src");
  videoPlayer.load();

  liveImg.classList.add("is-hidden");
  videoPlayer.classList.add("is-hidden");
  standbyEl.classList.remove("is-hidden");
  hideProgress();

  if (isWarning) {
    standbyEl.classList.add("is-warning");
    standbyIcon.innerHTML = `
      <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#ef4444" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
        <line x1="12" y1="9" x2="12" y2="13"/>
        <line x1="12" y1="17" x2="12.01" y2="17"/>
      </svg>
    `;
    standbyTitle.textContent = "NO VIDEO TO ANALYZE";
    standbyDesc.innerHTML =
      "Please <b>drag &amp; drop or select a video file (.mp4, .avi, .mov)</b> in the left control panel before starting!";
  } else {
    standbyEl.classList.remove("is-warning");
    standbyIcon.innerHTML = `
      <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
        <rect x="2" y="3" width="20" height="14" rx="2" ry="2"/>
        <line x1="8" y1="21" x2="16" y2="21"/>
        <line x1="12" y1="17" x2="12" y2="21"/>
      </svg>
    `;
    standbyTitle.textContent = "NO VIDEO LOADED";
    standbyDesc.innerHTML =
      "Drag and drop a dashcam video file (<b>.mp4</b>, <b>.avi</b>, <b>.mov</b>) into the left control panel to start analysis.";
  }
}

export function showPosterPreview(posterB64) {
  revokeBlobUrl();
  videoPlayer.pause();
  videoPlayer.classList.add("is-hidden");
  standbyEl.classList.add("is-hidden");
  liveImg.classList.remove("is-hidden");
  if (posterB64) {
    liveImg.src = `data:image/jpeg;base64,${posterB64}`;
  }
  hideProgress();
}

export function renderLiveFrameBlob(blob) {
  const nextUrl = URL.createObjectURL(blob);
  standbyEl.classList.add("is-hidden");
  videoPlayer.classList.add("is-hidden");
  liveImg.classList.remove("is-hidden");

  const prevUrl = currentBlobUrl;
  currentBlobUrl = nextUrl;
  liveImg.src = nextUrl;

  if (prevUrl) {
    URL.revokeObjectURL(prevUrl);
  }
}

export function showAnalyzedVideoPlayer(videoUrl) {
  revokeBlobUrl();
  standbyEl.classList.add("is-hidden");
  liveImg.classList.add("is-hidden");
  videoPlayer.classList.remove("is-hidden");
  videoPlayer.src = videoUrl;
  videoPlayer.play().catch(() => {});
}

export function updateProgress(currentFrame, totalFrames, fpsVal = 30.0, completed = false) {
  const safeTotal = Math.max(1, totalFrames || 1);
  const pct = completed ? 100 : Math.min(100, Math.floor((currentFrame / safeTotal) * 100));
  const safeFps = Math.max(1.0, fpsVal || 30.0);
  const curSec = Math.floor(currentFrame / safeFps);
  const totSec = Math.floor(safeTotal / safeFps);

  const fmt = (s) => `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
  progressBar.classList.remove("is-hidden");

  if (completed) {
    progressLabel.textContent = `Analysis Complete: ${safeTotal} / ${safeTotal} frames (100%)`;
    progressLabel.style.color = "#059669";
    progressFill.style.width = "100%";
    progressFill.style.background = "#10b981";
  } else {
    progressLabel.textContent = `Analysis Progress: Frame ${currentFrame} / ${safeTotal} (${pct}%)`;
    progressLabel.style.color = "#475569";
    progressFill.style.width = `${pct}%`;
    progressFill.style.background = "#2563eb";
  }
  progressMeta.textContent = `${fmt(curSec)} / ${fmt(totSec)}`;
}

export function hideProgress() {
  progressBar.classList.add("is-hidden");
}
