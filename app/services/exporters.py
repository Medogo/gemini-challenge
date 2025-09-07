# app/services/exporters.py
import io, csv, hashlib, datetime as dt
from typing import Dict, Any, List

def _now_utc_stamp() -> str:
    return dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

def _fmt_dt_local(iso: str) -> str:
    """
    Transforme 'YYYY-MM-DDTHH:MM' -> 'YYYYMMDDTHHMM00' (heure flottante sans TZ).
    Si l'ISO contient déjà des secondes, on les garde (en retirant les symboles).
    """
    if not iso:
        return ""
    # Basique: YYYY-MM-DDTHH:MM[:SS]
    date, time = iso.split("T")
    y, m, d = date.split("-")
    parts = time.split(":")
    hh, mm = parts[0], parts[1]
    ss = parts[2] if len(parts) > 2 else "00"
    return f"{y}{m}{d}T{hh}{mm}{ss}"

def build_ics(project_name: str, schedule_json: Dict[str, Any]) -> bytes:
    """
    Construit un .ics simple avec 1 VEVENT par slot.
    """
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//Social Creative Coach//EN",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
    ]
    dtstamp = _now_utc_stamp()

    for ch in schedule_json.get("schedule", []):
        channel = ch.get("channel", "Unknown")
        for slot in ch.get("slots", []):
            iso = slot.get("iso", "")
            why = slot.get("why", "")
            dtstart = _fmt_dt_local(iso)
            if not dtstart:
                continue
            uid_src = f"{project_name}|{channel}|{iso}|{why}"
            uid = hashlib.md5(uid_src.encode("utf-8")).hexdigest() + "@social-coach"

            summary = f"{project_name} — {channel}"
            desc = (why or "").replace("\n", "\\n")
            lines += [
                "BEGIN:VEVENT",
                f"UID:{uid}",
                f"DTSTAMP:{dtstamp}",
                f"DTSTART:{dtstart}",
                f"SUMMARY:{summary}",
                f"DESCRIPTION:{desc}",
                "END:VEVENT",
            ]

    lines += ["END:VCALENDAR", ""]
    return "\r\n".join(lines).encode("utf-8")

def build_csv(variants_json: Dict[str, Any]) -> bytes:
    """
    Construit un CSV: channel,title,body,hashtags,cta
    """
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["channel", "title", "body", "hashtags", "cta"])
    for v in variants_json.get("variants", []):
        ch = v.get("channel", "")
        title = v.get("title", "")
        body = v.get("body", "")
        tags = v.get("hashtags", []) or []
        cta = v.get("cta", "")
        w.writerow([ch, title, body, " ".join(f"#{t}" for t in tags), cta])
    return buf.getvalue().encode("utf-8")
