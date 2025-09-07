import io
import json
import zipfile
from datetime import datetime
from typing import Dict, Any, Optional, List

def _md_readme(project_name: str,
               variants_json: Dict[str, Any],
               schedule_json: Dict[str, Any],
               image_prompt_json: Dict[str, Any],
               brand_name: Optional[str],
               brand_color: Optional[str],
               credits: Optional[str]) -> str:
    lines: List[str] = []
    lines += [f"# {project_name}", ""]
    lines += ["_Kit de création généré automatiquement._", ""]
    if brand_name or brand_color:
        lines += ["## Marque", ""]
        if brand_name:  lines += [f"- **Nom** : {brand_name}"]
        if brand_color: lines += [f"- **Couleur dominante** : `{brand_color}`"]
        lines += [""]

    lines += ["## Prompt d’image", ""]
    lines += ["```", image_prompt_json.get("image_prompt",""), "```", ""]

    lines += ["## Variantes", ""]
    for v in variants_json.get("variants", []):
        lines += [
            f"### {v.get('channel','')}",
            f"**Titre** : {v.get('title','')}",
            "",
            v.get("body",""),
            "",
            "**Hashtags** : " + " ".join(f"#{t}" for t in (v.get("hashtags",[]) or [])),
            f"**CTA** : {v.get('cta','')}",
            ""
        ]

    lines += ["## Planning (7 jours)", ""]
    for ch in schedule_json.get("schedule", []):
        lines += [f"### {ch.get('channel','')}", ""]
        for slot in ch.get("slots", []):
            lines += [f"- {slot.get('iso','')} — {slot.get('why','')}", ""]
    lines += [""]

    lines += ["---", f"_Généré le {datetime.utcnow().isoformat(timespec='seconds')}Z_", ""]
    if credits:
        lines += [f"_Crédits_: {credits}", ""]

    return "\n".join(lines)

def build_zip_kit(project_name: str,
                  variants_json: Dict[str, Any],
                  schedule_json: Dict[str, Any],
                  image_prompt_json: Dict[str, Any],
                  brand_name: Optional[str] = None,
                  brand_color: Optional[str] = None,
                  credits: Optional[str] = "Social Creative Coach (Gemini)") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        # JSON bruts
        z.writestr("variants.json", json.dumps(variants_json, ensure_ascii=False, indent=2))
        z.writestr("schedule.json", json.dumps(schedule_json, ensure_ascii=False, indent=2))
        z.writestr("image_prompt.json", json.dumps(image_prompt_json, ensure_ascii=False, indent=2))
        # README
        z.writestr(
            "README.md",
            _md_readme(project_name, variants_json, schedule_json, image_prompt_json, brand_name, brand_color, credits)
        )
        # Petits .txt par canal
        for v in variants_json.get("variants", []):
            ch = v.get("channel","Unknown")
            base = f"posts/{ch}".replace(" ", "_")
            body = v.get("body","")
            title = v.get("title","")
            hashtags = " ".join(f"#{t}" for t in (v.get("hashtags",[]) or []))
            cta = v.get("cta","")
            z.writestr(f"{base}.txt", f"{title}\n\n{body}\n\n{hashtags}\n\nCTA: {cta}\n")
    return buf.getvalue()
