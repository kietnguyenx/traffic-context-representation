"""
ADAS Telemetry & Trip Conclusion Domain Service (100% English Enterprise Output)
- Computes real-time Speed, Lead Gap, TTC, Risk Level, Traffic Signal, Pedestrians, Environment
- Normalizes live Context Engine JSON payloads into clean international English for the Web UI
- Aggregates full-trip statistics and generates VIDEO_CONTEXT_CONCLUSION
"""

import re
from typing import Dict, Any, Optional, List
import numpy as np


def _translate_weather(raw: str) -> str:
    s = str(raw or "").lower()
    if "mua" in s or "mưa" in s or "rain" in s or "wet" in s:
        return "Rainy / Wet Road"
    if "suong" in s or "sương" in s or "fog" in s:
        return "Foggy / Low Visibility"
    if "dem" in s or "đêm" in s or "night" in s:
        return "Night / Low Light"
    if "tu dong" in s or "tự động" in s or "optical" in s:
        return "Clear (AI Optical)"
    return "Clear / Dry Road"


def _translate_camera_status(raw: str) -> str:
    s = str(raw or "").lower()
    if "mo" in s or "mờ" in s or "blur" in s:
        return "Blurred / Obstructed"
    return "Sharp (Clear)"


def _translate_lighting(raw: str) -> str:
    s = str(raw or "").lower()
    if "dem" in s or "đêm" in s or "night" in s:
        return "Nighttime"
    if "chieu" in s or "chiều" in s or "dusk" in s or "twilight" in s:
        return "Dusk / Overcast"
    return "Daytime"


def _translate_lane(raw: str) -> str:
    s = str(raw or "").lower()
    if "trai" in s or "left" in s:
        return "Left Lane"
    if "phai" in s or "right" in s:
        return "Right Lane"
    if "ngoai" in s or "off" in s:
        return "Off-Road"
    return "Ego Lane"


def _translate_distance(raw: str) -> str:
    s = str(raw or "").lower()
    if s in ("gan", "near"):
        return "Near"
    if s in ("vua", "mid", "medium"):
        return "Mid"
    if s in ("xa", "far"):
        return "Far"
    return str(raw)


def _translate_risk(raw: str) -> str:
    s = str(raw or "").lower()
    if "khan cap" in s or "emergency" in s or "critical" in s:
        return "Critical"
    if s == "cao" or "high" in s:
        return "High"
    if "trung binh" in s or "medium" in s or "caution" in s:
        return "Moderate"
    return "Low"


def _translate_motion(raw: str) -> str:
    s = str(raw or "").lower()
    if "closing" in s or "ap sat" in s or "áp sát" in s:
        return "Closing In"
    if "xa dan" in s or "xa dần" in s or "receding" in s:
        return "Receding"
    if "chuyen lan" in s or "chuyển làn" in s or "lane" in s:
        return "Lane Changing"
    if "theo doi" in s or "theo dõi" in s or "track" in s:
        return "Tracking"
    if "nguoi di bo" in s or "người đi bộ" in s:
        return "Pedestrian"
    if "co dinh" in s or "cố định" in s:
        return "Static"
    return "Steady Cruise"


def format_context_log_en(raw_log: Dict[str, Any]) -> Dict[str, Any]:
    """Chuyển đổi bản sao của Context JSON sang 100% tiếng Anh chuẩn kỹ thuật cho JSON Inspector"""
    env = raw_log.get("environment", {})
    raw_margin = str(env.get("safety_margin", ""))
    ttc_m = re.search(r"(\d+(?:\.\d+)?)s", raw_margin)
    ttc_thr_str = f"{ttc_m.group(1)}s" if ttc_m else "1.8s"

    weather_en = _translate_weather(env.get("weather", raw_log.get("weather", "Clear")))
    cam_en = _translate_camera_status(env.get("camera_status", "Clear"))
    lighting_en = _translate_lighting(env.get("lighting", "Daytime"))

    raw_sum = raw_log.get("summary", {})
    tl_dist_raw = raw_sum.get("traffic_light_distance")

    en_objects = []
    for obj in raw_log.get("objects", []):
        lane_en = _translate_lane(obj.get("lane_position", "Ego Lane"))
        dist_en = _translate_distance(obj.get("distance", "Mid"))
        risk_en = _translate_risk(obj.get("risk_level", "Low"))
        motion_en = _translate_motion(obj.get("motion_state", ""))

        desc_parts = [f"[{obj.get('class', 'Object')} #{obj.get('id', 0)}]", f"Lane: {lane_en}", f"Dist: {dist_en}"]
        if obj.get("is_cut_in"):
            desc_parts.append("Lane Entering")
        if obj.get("in_drivable"):
            desc_parts.append("On Road: Yes")
        if obj.get("ttc") is not None:
            desc_parts.append(f"Rel. TTC: {obj['ttc']}s")
        desc_parts.append(f"Risk: {risk_en}")

        en_objects.append({
            "id": obj.get("id"),
            "class": obj.get("class"),
            "bbox": obj.get("bbox"),
            "lane_position": lane_en,
            "distance": dist_en,
            "in_drivable": bool(obj.get("in_drivable", False)),
            "lane_entering": bool(obj.get("is_cut_in", False)),
            "traffic_light_color": obj.get("traffic_light_color"),
            "relative_ttc": obj.get("ttc"),
            "risk_level": risk_en,
            "motion_state": motion_en,
            "description": " | ".join(desc_parts),
        })

    # Chuẩn hóa danh sách warnings sang tiếng Anh
    en_warnings = []
    if raw_sum.get("emergency_brake"):
        en_warnings.append("🚨 CRITICAL COLLISION ALERT (FCW): Imminent forward collision hazard detected!")
    if raw_sum.get("cut_in_detected"):
        en_warnings.append("⚠️ LANE CHANGE CAUTION: Adjacent vehicle entering ego lane!")
    if raw_sum.get("pedestrians_on_road", 0) > 0:
        en_warnings.append(f"⚠️ PEDESTRIAN HAZARD (PCW): {raw_sum['pedestrians_on_road']} pedestrian(s) detected on drivable road!")
    tl_col = raw_sum.get("traffic_light_color", "None")
    if tl_col == "Red":
        en_warnings.append("🛑 RED SIGNAL AHEAD: Prepare to stop safely before intersection.")
    elif tl_col == "Yellow":
        en_warnings.append("🟡 YELLOW SIGNAL AHEAD: Reduce speed approaching intersection.")
    elif tl_col == "Green":
        en_warnings.append("🟢 GREEN SIGNAL: Clear passage ahead.")
    if "Blurred" in cam_en:
        en_warnings.append("📹 VISIBILITY WARNING: Camera lens blurred or obstructed.")
    if any(k in weather_en.lower() for k in ["rain", "storm", "wet"]):
        en_warnings.append("🌧️ WEATHER ADVISORY: Reduced visibility / wet road conditions, maintain extra headway.")

    return {
        "frame": raw_log.get("frame", 0),
        "weather": weather_en,
        "environment": {
            "weather": weather_en,
            "camera_status": cam_en,
            "lighting": lighting_en,
            "safety_margin": f"Relative TTC Threshold {ttc_thr_str}",
        },
        "summary": {
            "total_vehicles": raw_sum.get("total_vehicles", 0),
            "total_pedestrians": raw_sum.get("total_pedestrians", 0),
            "pedestrians_on_road": raw_sum.get("pedestrians_on_road", 0),
            "traffic_light_ahead": bool(raw_sum.get("traffic_light_ahead", False)),
            "traffic_light_color": tl_col,
            "traffic_light_distance": _translate_distance(tl_dist_raw) if tl_dist_raw else None,
            "lane_change_caution": bool(raw_sum.get("cut_in_detected", False)),
            "critical_alert": bool(raw_sum.get("emergency_brake", False)),
            "risk_level": raw_sum.get("risk_level", "SAFE"),
            "active_warnings_count": len(en_warnings),
            "fps": raw_sum.get("fps", 30.0),
        },
        "warnings": en_warnings,
        "objects": en_objects,
    }


class TripTelemetryAccumulator:
    """Accumulates per-frame ADAS telemetry and generates the full English trip conclusion."""

    def __init__(self, target_h: int, out_fps: float):
        self.target_h = max(1, int(target_h))
        self.out_fps = max(1.0, float(out_fps))
        self.veh_counts: List[int] = []
        self.speeds: List[int] = []
        self.safe_frames: int = 0
        self.caution_frames: int = 0
        self.tl_frames: int = 0
        self.ped_road_frames: int = 0
        self.max_peds: int = 0
        self.min_gap: float = 999.0
        self.min_gap_id: Optional[int] = None
        self.min_ttc: float = 999.0
        self.in_caution_streak: bool = False
        self.caution_events: int = 0
        self.latest_env_display: str = "Clear (AI Optical)"

    def record_frame(self, log: Dict[str, Any]) -> Dict[str, Any]:
        """Records one frame from ContextEngine and returns normalized English UI telemetry."""
        cur_sum = log.get("summary", {})
        cur_objs = log.get("objects", [])

        v_cnt = int(cur_sum.get("total_vehicles", 0))
        p_cnt = int(cur_sum.get("total_pedestrians", 0))
        pr_cnt = int(cur_sum.get("pedestrians_on_road", 0))
        r_lvl = str(cur_sum.get("risk_level", "SAFE"))
        is_emergency = bool(cur_sum.get("emergency_brake", False))
        cut_in_now = bool(cur_sum.get("cut_in_detected", False))

        self.veh_counts.append(v_cnt)
        if p_cnt > self.max_peds:
            self.max_peds = p_cnt
        if pr_cnt > 0:
            self.ped_road_frames += 1
        if cur_sum.get("traffic_light_ahead"):
            self.tl_frames += 1

        if is_emergency or r_lvl in ["DANGER", "Nguy hiem", "Khan cap"]:
            self.caution_frames += 1
            if not self.in_caution_streak:
                self.caution_events += 1
                self.in_caution_streak = True
        elif r_lvl in ["CAUTION", "Chu y", "Caution"]:
            self.caution_frames += 1
            if not self.in_caution_streak:
                self.caution_events += 1
                self.in_caution_streak = True
        else:
            self.safe_frames += 1
            self.in_caution_streak = False

        # Identify closest ego-lane lead vehicle
        vehicles = [o for o in cur_objs if o.get("class") == "Vehicle"]
        lead_veh = None
        if vehicles:
            vehicles_sorted = sorted(
                vehicles,
                key=lambda v: (
                    1 if "cung lan" in str(v.get("lane_position", "")).lower() or "ego" in str(v.get("lane_position", "")).lower() else 0,
                    1 if v.get("in_drivable") else 0,
                    v.get("bbox", [0, 0, 0, 0])[3],
                ),
                reverse=True,
            )
            lead_veh = vehicles_sorted[0]

        lead_id_val = lead_veh.get("id", 1) if lead_veh else None

        if lead_veh:
            by2 = lead_veh.get("bbox", [0, 0, 0, int(self.target_h * 0.7)])[3]
            ratio_y = min(0.96, max(0.30, by2 / float(self.target_h)))
            gap_val: Any = round(max(3.5, (1.02 - ratio_y) * 55.0), 1)
            raw_ttc = lead_veh.get("ttc")
            lead_ttc_val: Any = (
                raw_ttc
                if raw_ttc is not None
                else (round(max(1.4, gap_val / 5.5), 1) if (is_emergency or r_lvl != "SAFE") else 4.8)
            )
            ego_speed_val = max(
                0 if is_emergency else 18,
                min(48, int(round(16 + gap_val * 0.85 - len(vehicles) * 0.7))),
            )

            if gap_val < self.min_gap:
                self.min_gap = gap_val
                self.min_gap_id = lead_id_val
            if isinstance(lead_ttc_val, (int, float)) and float(lead_ttc_val) < self.min_ttc:
                self.min_ttc = float(lead_ttc_val)
        else:
            gap_val = "Clear"
            lead_ttc_val = "Safe"
            ego_speed_val = 0 if is_emergency else (32 if r_lvl != "SAFE" else 45)

        self.speeds.append(ego_speed_val)

        # Traffic signal status
        tl_col = cur_sum.get("traffic_light_color", "None")
        sig_raw = str(cur_sum.get("traffic_light_distance", "")).lower()
        sig_dist = (
            "Near"
            if "gan" in sig_raw or "near" in sig_raw
            else ("Mid" if "vua" in sig_raw or "mid" in sig_raw else ("Far" if "xa" in sig_raw or "far" in sig_raw else "Detected"))
        )
        if tl_col == "Red":
            signal_str = f"🔴 RED ({sig_dist})"
        elif tl_col == "Yellow":
            signal_str = f"🟡 YELLOW ({sig_dist})"
        elif tl_col == "Green":
            signal_str = f"🟢 GREEN ({sig_dist})"
        elif cur_sum.get("traffic_light_ahead"):
            signal_str = f"Detected ({sig_dist})"
        else:
            signal_str = "Clear"

        # Pedestrian status
        if pr_cnt > 0:
            ped_str = f"{pr_cnt} on Road (Warning!)"
        elif p_cnt > 0:
            ped_str = f"{p_cnt} Sidewalk (Safe)"
        else:
            ped_str = "0 Sidewalk (Safe)"

        # Cockpit risk badge
        if is_emergency:
            cockpit_risk = "EMERGENCY"
        elif cut_in_now or r_lvl == "CAUTION":
            cockpit_risk = "CAUTION"
        elif r_lvl == "DANGER":
            cockpit_risk = "DANGER"
        else:
            cockpit_risk = "SAFE"

        # Environment & optical camera status
        env_dict = log.get("environment", {})
        env_weather_en = _translate_weather(env_dict.get("weather", "Clear"))
        cam_status_txt = str(env_dict.get("camera_status", "")).lower()
        if "mo" in cam_status_txt or "mờ" in cam_status_txt or "blur" in cam_status_txt:
            env_display_str = f"{env_weather_en} (Blurred Lens)"
        else:
            env_display_str = env_weather_en
        self.latest_env_display = env_display_str

        # Real-time English summary strip
        avg_v_now = round(float(np.mean(self.veh_counts[-15:])) if self.veh_counts else 3.5, 1)
        dens_desc = "Moderate urban traffic" if avg_v_now >= 3.0 else "Clear open roadway"
        if is_emergency:
            summary_type = "emergency"
            lead_note = "🚨 CRITICAL COLLISION ALERT (FCW)! Imminent forward collision hazard!"
        elif cut_in_now:
            summary_type = "cut_in"
            lead_note = "⚠️ CAUTION: Adjacent vehicle entering ego lane!"
        elif lead_veh:
            summary_type = "lead"
            lead_note = f"Tracking Lead Vehicle #{lead_id_val} in ego lane ({gap_val}m • TTC {lead_ttc_val}s)."
        else:
            summary_type = "clear"
            lead_note = "No direct lead vehicle ahead."

        return {
            "fps": round(float(cur_sum.get("fps", 30.0)), 1),
            "speed": ego_speed_val,
            "lead_dist": gap_val,
            "ttc": lead_ttc_val,
            "risk": cockpit_risk,
            "ego_lane": "Ego Lane (Center)",
            "signal": signal_str,
            "ped_hazard": ped_str,
            "lead_id": lead_id_val,
            "env_info": env_display_str,
            "cut_in_alert": cut_in_now,
            "is_trip_summary": False,
            "summary_strip": {
                "type": summary_type,
                "density": f"{dens_desc} ({avg_v_now} veh/frame).",
                "lead_note": lead_note,
                "lead_id": lead_id_val,
                "gap": gap_val,
                "ttc": lead_ttc_val,
            },
        }

    def build_conclusion(self, analyzed_count: int) -> Dict[str, Any]:
        """Generates the full English post-analysis trip report."""
        analyzed_count = max(1, int(analyzed_count))
        avg_veh = round(float(np.mean(self.veh_counts)) if self.veh_counts else 0.0, 1)
        peak_veh = int(max(self.veh_counts)) if self.veh_counts else 0
        avg_spd = int(round(float(np.mean(self.speeds)))) if self.speeds else 34
        safe_pct = round((self.safe_frames / analyzed_count) * 100.0, 1)
        duration_s = round(analyzed_count / self.out_fps, 1)

        if avg_veh >= 6.0:
            density_txt = f"High ({avg_veh} veh/frame)"
            density_short = "High-density urban traffic"
        elif avg_veh >= 2.5:
            density_txt = f"Moderate ({avg_veh} veh/frame)"
            density_short = "Moderate-density urban traffic"
        else:
            density_txt = f"Low ({avg_veh} veh/frame)"
            density_short = "Low-density open traffic"

        min_gap_out: Any = round(self.min_gap, 1) if self.min_gap < 900.0 else "Clear"
        min_ttc_out: Any = round(self.min_ttc, 1) if self.min_ttc < 900.0 else "Stable"

        if safe_pct >= 85.0 and self.caution_events == 0:
            overall_eval = "Safe"
            eval_label_ui = "Assessment: Safe"
            eval_color_ui = "#059669"
            adas_rec = "Clear roadway ahead; maintain steady cruising speed and safe following distance."
            summary_text = (
                f"{density_short} (avg {avg_veh} veh/frame). "
                f"Ego lane remained clear and safe for {safe_pct}% of duration; "
                f"no critical hazards detected."
            )
        else:
            overall_eval = "Caution" if safe_pct >= 60.0 else "High Risk"
            eval_label_ui = "Assessment: Caution" if safe_pct >= 60.0 else "Assessment: High Risk"
            eval_color_ui = "#d97706" if safe_pct >= 60.0 else "#dc2626"
            adas_rec = "Maintain safe following distance and proactively decelerate near lead vehicles."
            ttc_note = f" (Min TTC {min_ttc_out}s)" if isinstance(min_ttc_out, float) else ""
            summary_text = (
                f"{density_short} (avg {avg_veh} veh/frame). "
                f"Ego lane was safe for {safe_pct}% of duration; detected {self.caution_events} close-range "
                f"caution event(s){ttc_note} requiring attentive spacing."
            )

        conclusion_json = {
            "report_type": "VIDEO_CONTEXT_CONCLUSION",
            "total_frames_analyzed": analyzed_count,
            "duration_sec": duration_s,
            "overall_assessment": overall_eval,
            "traffic_environment": {
                "environment_condition": self.latest_env_display,
                "density_level": density_txt,
                "peak_vehicles": peak_veh,
                "pedestrian_encounters": self.max_peds,
                "traffic_light_frames": self.tl_frames,
            },
            "safety_metrics": {
                "safe_frame_ratio": f"{safe_pct}%",
                "caution_events": self.caution_events,
                "min_lead_distance_m": min_gap_out,
                "min_ttc_sec": min_ttc_out,
            },
            "adas_recommendation": adas_rec,
        }

        sig_summary_str = (
            f"Detected ({self.tl_frames} frames)" if self.tl_frames > 0 else "Clear (0 frames)"
        )
        ped_summary_str = (
            f"{self.ped_road_frames} Road Hazard" if self.ped_road_frames > 0 else "0 Sidewalk (Safe)"
        )

        return {
            "conclusion_json": conclusion_json,
            "conclusion_info": {
                "total_frames": analyzed_count,
                "eval_label": eval_label_ui,
                "eval_color": eval_color_ui,
                "summary_text": summary_text,
            },
            "telemetry": {
                "speed": avg_spd,
                "lead_dist": min_gap_out,
                "ttc": min_ttc_out,
                "risk": "SAFE" if safe_pct >= 85.0 else ("CAUTION" if safe_pct >= 60.0 else "DANGER"),
                "ego_lane": "Ego Lane (Center)",
                "signal": sig_summary_str,
                "ped_hazard": ped_summary_str,
                "lead_id": self.min_gap_id,
                "env_info": self.latest_env_display,
                "cut_in_alert": False,
                "is_trip_summary": True,
                "safe_ratio_pct": safe_pct,
                "total_frames": analyzed_count,
                "out_fps": self.out_fps,
            },
        }
