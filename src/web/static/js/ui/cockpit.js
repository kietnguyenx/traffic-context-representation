/**
 * Component Cụm Đồng Hồ Buồng Lái ADAS Dạng Băng Ngang (Ribbon Dock)
 * Cập nhật DOM chính xác theo từng phần tử thay vì vẽ lại toàn bộ HTML -> Không nháy, siêu mượt.
 */

const dom = {
  lblSpeed: document.getElementById("lbl-speed"),
  tagSpeed: document.getElementById("tag-speed"),
  valSpeed: document.getElementById("val-speed"),
  meterSpeed: document.getElementById("meter-speed"),

  lblLead: document.getElementById("lbl-lead"),
  tagLead: document.getElementById("tag-lead"),
  valGap: document.getElementById("val-gap"),
  meterGap: document.getElementById("meter-gap"),

  lblTtc: document.getElementById("lbl-ttc"),
  valTtc: document.getElementById("val-ttc"),
  wrapTtc: document.getElementById("wrap-ttc"),
  meterTtc: document.getElementById("meter-ttc"),

  lblRisk: document.getElementById("lbl-risk"),
  valRisk: document.getElementById("val-risk"),
  meterRisk: document.getElementById("meter-risk"),

  subLane: document.getElementById("sub-lane"),
  subSignal: document.getElementById("sub-signal"),
  subPed: document.getElementById("sub-ped"),
  subEnv: document.getElementById("sub-env"),
};

export function renderCockpitTelemetry({
  speed = "--",
  lead_dist = "--",
  ttc = "--",
  risk = "STANDBY",
  ego_lane = "Ego Lane (Center)",
  signal = "Standby",
  ped_hazard = "0 Sidewalk (Safe)",
  lead_id = null,
  env_info = "Clear",
  cut_in_alert = false,
  is_trip_summary = false,
  safe_ratio_pct = 100.0,
} = {}) {
  // 1. Speed
  dom.lblSpeed.textContent = is_trip_summary ? "AVG SPEED" : "CURRENT SPEED";
  dom.tagSpeed.textContent = is_trip_summary ? "AVERAGE" : "NOMINAL";
  dom.valSpeed.textContent = String(speed);
  const speedPct =
    typeof speed === "number" ? Math.min(100, Math.max(10, Math.round((speed / 80.0) * 100))) : 0;
  dom.meterSpeed.style.width = `${speedPct}%`;

  // 2. Lead Vehicle Gap
  dom.lblLead.textContent = is_trip_summary ? "MIN LEAD GAP" : "LEAD VEHICLE GAP";
  const tagLeadText = cut_in_alert ? "⚠️ CUT-IN!" : lead_id != null ? "#1 EGO LANE" : "CLEAR";
  const tagLeadColor = cut_in_alert ? "#dc2626" : "#0284c7";
  dom.tagLead.textContent = tagLeadText;
  dom.tagLead.style.color = tagLeadColor;
  dom.valGap.textContent = String(lead_dist);
  const gapPct =
    typeof lead_dist === "number"
      ? Math.min(100, Math.max(12, Math.round((lead_dist / 45.0) * 100)))
      : ["clear", "trống"].includes(String(lead_dist).toLowerCase())
      ? 100
      : 0;
  dom.meterGap.style.width = `${gapPct}%`;
  dom.meterGap.style.background = tagLeadColor;

  // 3. Time To Collision (TTC)
  dom.lblTtc.textContent = is_trip_summary ? "MIN TTC" : "TIME TO COLLISION";
  dom.valTtc.textContent = String(ttc);

  let ttcColor = "#0f172a";
  let ttcBarColor = "#94a3b8";
  let ttcPct = 0;

  if (typeof ttc === "number") {
    if (ttc < 1.3) {
      ttcColor = "#dc2626";
      ttcBarColor = "#dc2626";
    } else if (ttc < 3.0) {
      ttcColor = "#d97706";
      ttcBarColor = "#f59e0b";
    } else {
      ttcColor = "#059669";
      ttcBarColor = "#10b981";
    }
    ttcPct = Math.min(100, Math.max(15, Math.round((ttc / 6.0) * 100)));
  } else if (["safe", "stable", "an toàn", "ổn định"].includes(String(ttc).toLowerCase())) {
    ttcColor = "#059669";
    ttcBarColor = "#10b981";
    ttcPct = 85;
  }
  dom.wrapTtc.style.color = ttcColor;
  dom.meterTtc.style.width = `${ttcPct}%`;
  dom.meterTtc.style.background = ttcBarColor;

  // 4. Risk Assessment
  dom.lblRisk.textContent = is_trip_summary ? "TRIP ASSESSMENT" : "RISK ASSESSMENT";
  let riskColor = "#64748b";
  let riskBadge = "○ STANDBY";
  let riskBarColor = "#94a3b8";
  let riskPct = 0;

  if (is_trip_summary) {
    riskBadge = `● ${Number(safe_ratio_pct).toFixed(1)}% SAFE`;
    riskPct = Math.round(safe_ratio_pct);
    if (safe_ratio_pct >= 85.0) {
      riskColor = "#059669";
      riskBarColor = "#10b981";
    } else if (safe_ratio_pct >= 60.0) {
      riskColor = "#d97706";
      riskBarColor = "#f59e0b";
    } else {
      riskColor = "#dc2626";
      riskBarColor = "#dc2626";
    }
  } else {
    const rUpper = String(risk).toUpperCase();
    if (["EMERGENCY", "AEB", "KHAN CAP", "PHANH"].some((k) => rUpper.includes(k))) {
      riskColor = "#dc2626";
      riskBadge = "🚨 AEB ACTIVE";
      riskBarColor = "#dc2626";
      riskPct = 100;
    } else if (["CAUTION", "CHU Y", "WARN", "WARNING"].includes(rUpper)) {
      riskColor = "#d97706";
      riskBadge = "● CAUTION";
      riskBarColor = "#f59e0b";
      riskPct = 68;
    } else if (["DANGER", "NGUY HIEM", "CRITICAL"].includes(rUpper)) {
      riskColor = "#dc2626";
      riskBadge = "● DANGER";
      riskBarColor = "#dc2626";
      riskPct = 92;
    } else if (["SAFE", "AN TOAN"].includes(rUpper)) {
      riskColor = "#059669";
      riskBadge = "● SAFE";
      riskBarColor = "#10b981";
      riskPct = 25;
    }
  }
  dom.valRisk.textContent = riskBadge;
  dom.valRisk.style.color = riskColor;
  dom.meterRisk.style.width = `${riskPct}%`;
  dom.meterRisk.style.background = riskBarColor;

  // 5. Sub-status Bar
  dom.subLane.textContent = ego_lane;

  const sigU = String(signal).toUpperCase();
  let sigColor = "#64748b";
  if (sigU.includes("RED") || sigU.includes("ĐỎ")) sigColor = "#dc2626";
  else if (sigU.includes("YELLOW") || sigU.includes("VÀNG") || sigU.includes("AMBER")) sigColor = "#d97706";
  else if (sigU.includes("GREEN") || sigU.includes("XANH")) sigColor = "#059669";
  dom.subSignal.textContent = signal;
  dom.subSignal.style.color = sigColor;

  const pedU = String(ped_hazard).toUpperCase();
  const isPedHazard =
    ["WARN", "HAZARD", "LÒNG ĐƯỜNG", "ROAD"].some((k) => pedU.includes(k)) &&
    !String(ped_hazard).startsWith("0");
  dom.subPed.textContent = ped_hazard;
  dom.subPed.style.color = isPedHazard ? "#dc2626" : "#059669";

  const envU = String(env_info).toUpperCase();
  let envColor = "#059669";
  if (["BLUR", "STORM", "MỜ", "BÃO"].some((k) => envU.includes(k))) envColor = "#dc2626";
  else if (["RAIN", "FOG", "NIGHT", "WET", "LOW LIGHT", "MƯA", "SƯƠNG", "ĐÊM"].some((k) => envU.includes(k))) envColor = "#d97706";
  dom.subEnv.textContent = env_info;
  dom.subEnv.style.color = envColor;
}
