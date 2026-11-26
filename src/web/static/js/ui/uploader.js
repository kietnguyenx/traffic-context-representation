/**
 * Component Quản Lý Khung Kéo-Thả Video & Thanh File Sẵn Sàng [✕]
 * Sử dụng HTML5 Native File Input + Drag & Drop (0ms response, không bao giờ lỗi kẹt).
 */

const fileInput = document.getElementById("video-file-input");
const dropzone = document.getElementById("video-dropzone");
const readyBar = document.getElementById("video-ready-bar");
const readyFileName = document.getElementById("ready-file-name");
const readyFileBadge = document.getElementById("ready-file-badge");
const clearBtn = document.getElementById("btn-clear-video");
const noVideoAlert = document.getElementById("no-video-alert");

export function initUploader({ onFileSelected, onClearClicked }) {
  // Nhấn vào khung dropzone -> mở hộp thoại chọn file gốc
  dropzone.addEventListener("click", () => {
    fileInput.value = "";
    fileInput.click();
  });

  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.value = "";
      fileInput.click();
    }
  });

  // Chọn file từ hộp thoại
  fileInput.addEventListener("change", () => {
    const file = fileInput.files && fileInput.files[0];
    if (file) {
      onFileSelected(file);
    }
  });

  // Kéo & Thả trực tiếp video
  const preventDefaults = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  ["dragenter", "dragover"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      preventDefaults(e);
      if (e.dataTransfer) e.dataTransfer.dropEffect = "copy";
      dropzone.classList.add("drag-over");
    });
  });

  ["dragleave", "dragend"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      preventDefaults(e);
      dropzone.classList.remove("drag-over");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    preventDefaults(e);
    dropzone.classList.remove("drag-over");
    const file = e.dataTransfer?.files?.[0];
    if (file) {
      onFileSelected(file);
    }
  });

  // Nút [✕] xóa video tức thì
  clearBtn.addEventListener("click", (e) => {
    e.preventDefault();
    e.stopPropagation();
    fileInput.value = "";
    onClearClicked();
  });
}

export function setUploaderEmpty() {
  fileInput.value = "";
  dropzone.classList.remove("is-hidden");
  readyBar.classList.add("is-hidden");
}

export function setUploaderUploading(fileName) {
  dropzone.classList.add("is-hidden");
  readyBar.classList.remove("is-hidden");
  readyFileName.textContent = fileName;
  readyFileName.title = fileName;
  readyFileBadge.textContent = "Uploading...";
  hideNoVideoAlert();
}

export function setUploaderReady(fileName) {
  dropzone.classList.add("is-hidden");
  readyBar.classList.remove("is-hidden");
  readyFileName.textContent = fileName;
  readyFileName.title = fileName;
  readyFileBadge.textContent = "Ready";
  hideNoVideoAlert();
}

export function showNoVideoAlert() {
  noVideoAlert.classList.remove("is-hidden");
}

export function hideNoVideoAlert() {
  noVideoAlert.classList.add("is-hidden");
}
