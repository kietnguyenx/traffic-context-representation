/**
 * Component Thông Báo Toast chuẩn Enterprise
 */

const container = document.getElementById("toast-container");

export function showToast(message, type = "info", durationMs = 3400) {
  if (!container) return;
  const item = document.createElement("div");
  item.className = `toast-item toast-${type}`;

  const iconMap = {
    success: "✅",
    error: "🚨",
    warning: "⚠️",
    info: "ℹ️",
  };
  item.textContent = `${iconMap[type] || "ℹ️"} ${message}`;
  container.appendChild(item);

  window.setTimeout(() => {
    item.style.opacity = "0";
    item.style.transform = "translateY(8px)";
    item.style.transition = "all 0.2s ease";
    window.setTimeout(() => item.remove(), 220);
  }, durationMs);
}
