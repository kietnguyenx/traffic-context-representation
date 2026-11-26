/**
 * Component Nhật Ký Ngữ Cảnh Thời Gian Thực (JSON Inspector chuẩn GitHub Light)
 * Escape XSS toàn bộ dữ liệu trước khi tô màu cú pháp JSON.
 */

const statusBadge = document.getElementById("json-status-badge");
const summaryStrip = document.getElementById("json-summary-strip");
const lineNumsEl = document.getElementById("json-line-nums");
const codeContentEl = document.getElementById("json-code-content");

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function highlightJsonLine(line) {
  let escaped = escapeHtml(line);
  escaped = escaped.replace(/(&quot;[^&]+?&quot;)(\s*:)/g, '<span class="jk">$1</span>$2');
  escaped = escaped.replace(/(:\s*)(&quot;[^&]*?&quot;)/g, '$1<span class="js">$2</span>');
  escaped = escaped.replace(/^(\s*)(&quot;[^&]*?&quot;)(,?)$/g, '$1<span class="js">$2</span>$3');
  escaped = escaped.replace(/(:\s*)(true|false|null)\b/g, '$1<span class="jb">$2</span>');
  escaped = escaped.replace(/(:\s*)(-?\d+(?:\.\d+)?)/g, '$1<span class="jn">$2</span>');
  return escaped;
}

export function renderJsonLog(jsonData, { summaryStripObj = null, conclusionInfo = null } = {}) {
  // 1. Badge & Summary Strip
  if (conclusionInfo) {
    statusBadge.className = "card-badge-done";
    statusBadge.textContent = `COMPLETED • ${conclusionInfo.total_frames || 0} FRAMES`;

    const evalColor = conclusionInfo.eval_color || "#059669";
    summaryStrip.classList.remove("is-hidden");
    summaryStrip.style.borderLeftColor = evalColor;
    summaryStrip.innerHTML =
      `<div class="summary-strip-title" style="color:${escapeHtml(evalColor)};">` +
      `TRIP CONTEXT CONCLUSION • ${escapeHtml(conclusionInfo.eval_label || "")}` +
      `</div><div>${escapeHtml(conclusionInfo.summary_text || "")}</div>`;
  } else {
    statusBadge.className = "frame-pill";
    const frameIdx = jsonData?.frame_index ?? jsonData?.frame;
    if (frameIdx !== undefined && frameIdx !== null) {
      statusBadge.textContent = `FRAME #${frameIdx}`;
    } else {
      statusBadge.textContent = String(jsonData?.status || "STANDBY");
    }

    if (summaryStripObj) {
      summaryStrip.classList.remove("is-hidden");
      summaryStrip.style.borderLeftColor =
        summaryStripObj.type === "emergency"
          ? "#dc2626"
          : summaryStripObj.type === "cut_in"
          ? "#d97706"
          : "#2563eb";

      const noteColor =
        summaryStripObj.type === "emergency"
          ? "#dc2626"
          : summaryStripObj.type === "cut_in"
          ? "#d97706"
          : "";

      const noteHtml = noteColor
        ? `<b style="color:${noteColor};">${escapeHtml(summaryStripObj.lead_note)}</b>`
        : escapeHtml(summaryStripObj.lead_note);

      summaryStrip.innerHTML = `${escapeHtml(summaryStripObj.density)} ${noteHtml}`;
    } else {
      summaryStrip.classList.add("is-hidden");
      summaryStrip.textContent = "";
    }
  }

  // 2. Format JSON (gom mảng tọa độ 4 số về 1 dòng giống bản chuẩn)
  let formatted = JSON.stringify(jsonData, null, 2);
  formatted = formatted.replace(
    /\[\s+(-?\d+(?:\.\d+)?),\s+(-?\d+(?:\.\d+)?),\s+(-?\d+(?:\.\d+)?),\s+(-?\d+(?:\.\d+)?)\s+\]/g,
    "[$1, $2, $3, $4]"
  );

  const lines = formatted.split("\n");
  lineNumsEl.innerHTML = lines.map((_, i) => String(i + 1)).join("<br>");
  codeContentEl.innerHTML = lines.map((l) => highlightJsonLine(l)).join("\n");
}
