/**
 * REST API Client cho Backend FastAPI
 */

export async function fetchSystemStatus() {
  const res = await fetch("/api/status", { cache: "no-store" });
  if (!res.ok) throw new Error("Unable to connect to ADAS server.");
  return res.json();
}

export async function uploadVideoFile(file) {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch("/api/video/upload", {
    method: "POST",
    body: formData,
  });
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.detail || "Video upload failed.");
  }
  return data;
}

export async function clearVideoOnServer() {
  const res = await fetch("/api/video/clear", { method: "POST" });
  if (!res.ok) throw new Error("Unable to clear video on server.");
  return res.json();
}

export async function syncTogglesToServer(toggles) {
  const res = await fetch("/api/toggles", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(toggles),
  });
  if (!res.ok) throw new Error("Unable to sync overlay layer settings.");
  return res.json();
}

export async function stopAnalysisOnServer() {
  await fetch("/api/analysis/stop", { method: "POST" }).catch(() => {});
}
