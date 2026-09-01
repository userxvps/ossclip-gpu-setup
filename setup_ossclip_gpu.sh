#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"

echo "🚀 Setting up OSSClip with Full Tesla T4 GPU Acceleration..."
nvidia-smi -pm 1 2>/dev/null || true

# 1. Upgrade Node.js to v22 LTS if needed
CURRENT_NODE=$(node -v 2>/dev/null || echo "v0")
if [[ ! "$CURRENT_NODE" =~ ^v22 ]]; then
  echo "📦 Installing Node.js v22 LTS..."
  curl -fsSL https://nodejs.org/dist/v22.23.2/node-v22.23.2-linux-x64.tar.xz -o /tmp/node22.tar.xz
  tar -xf /tmp/node22.tar.xz -C /tools
  rm -rf /tools/node
  mv /tools/node-v22.23.2-linux-x64 /tools/node
  export PATH="/tools/node/bin:$PATH"
fi

# 2. Install ossclip globally
echo "📦 Installing ossclip..."
npm install -g ossclip --silent

# 3. Install GPU Whisper & CTranslate2
echo "📦 Setting up Whisper GPU CUDA..."
pip install -q faster-whisper ctranslate2 gdown cairosvg pillow

# 4. Create whisper-cli GPU wrapper
cat << 'PYEOF' > /usr/local/bin/whisper-cli-gpu
#!/usr/bin/env python3
import sys, os, json, argparse
from faster_whisper import WhisperModel, BatchedInferencePipeline

def main():
    if "--help" in sys.argv or "-h" in sys.argv:
        print("usage: whisper-cli-gpu [options] <audio>")
        sys.exit(0)

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-m", "--model", default="small.en")
    parser.add_argument("-f", "--file")
    parser.add_argument("-oj", "--output-json", action="store_true")
    parser.add_argument("-of", "--output-file")
    parser.add_argument("-ml", "--max-len", type=int)
    parser.add_argument("-l", "--language", default="en")
    parser.add_argument("--prompt")
    parser.add_argument("-tr", "--translate", action="store_true")

    args, unknown = parser.parse_known_args()
    audio_path = args.file
    if not audio_path:
        for u in unknown:
            if not u.startswith("-") and os.path.exists(u):
                audio_path = u
                break

    if not audio_path or not os.path.exists(audio_path):
        print("Error: No audio file provided", file=sys.stderr)
        sys.exit(1)

    env_model = os.environ.get("WHISPER_GPU_MODEL", "").strip()
    if env_model:
        model_name = env_model
    else:
        model_name = "small.en"
        if args.model and "base" in args.model.lower():
            model_name = "base.en" if args.language == "en" else "base"
        elif args.model and "medium" in args.model.lower():
            model_name = "medium.en" if args.language == "en" else "medium"
        elif args.model and "large" in args.model.lower():
            model_name = "large-v3"

    batch_size = int(os.environ.get("WHISPER_BATCH_SIZE", "24"))

    device = "cuda"
    compute_type = "float16"
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() == 0:
            device = "cpu"
            compute_type = "int8"
    except Exception:
        device = "cpu"
        compute_type = "int8"

    try:
        if device == "cuda":
            model = WhisperModel(
                model_name,
                device="cuda",
                compute_type="float16",
                num_workers=2,
                cpu_threads=4
            )
        else:
            model = WhisperModel(model_name, device="cpu", compute_type="int8")
    except Exception as e:
        if device != "cpu":
            print(f"Warning: CUDA initialization failed ({e}). Falling back to CPU...", file=sys.stderr)
            device = "cpu"
            compute_type = "int8"
            model = WhisperModel(model_name, device="cpu", compute_type="int8")
        else:
            raise

    task = "translate" if args.translate else "transcribe"

    segments = None
    info = None
    if device == "cuda":
        try:
            batched_model = BatchedInferencePipeline(model=model)
            segments, info = batched_model.transcribe(
                audio_path,
                batch_size=batch_size,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=400),
                language=args.language if args.language != "auto" else None,
                task=task,
                initial_prompt=args.prompt,
                word_timestamps=True
            )
        except Exception as batch_err:
            print(f"Notice: Batched inference fallback ({batch_err}), using sequential CUDA...", file=sys.stderr)
            segments = None

    if segments is None:
        segments, info = model.transcribe(
            audio_path,
            language=args.language,
            task=task,
            initial_prompt=args.prompt,
            vad_filter=True,
            word_timestamps=True
        )

    detected_lang = getattr(info, "language", None) or (args.language if args.language and args.language != "auto" else "en")

    transcription_list = []
    for s in segments:
        words = s.words or []
        if words:
            for w in words:
                transcription_list.append({
                    "timestamps": {
                        "from": f"{format_whisper_time(w.start)}",
                        "to": f"{format_whisper_time(w.end)}"
                    },
                    "offsets": {
                        "from": int(w.start * 1000),
                        "to": int(w.end * 1000)
                    },
                    "text": w.word
                })
        else:
            transcription_list.append({
                "timestamps": {
                    "from": f"{format_whisper_time(s.start)}",
                    "to": f"{format_whisper_time(s.end)}"
                },
                "offsets": {
                    "from": int(s.start * 1000),
                    "to": int(s.end * 1000)
                },
                "text": s.text.strip()
            })

    output_data = {
        "systeminfo": "faster-whisper GPU Tensor Core Batched Acceleration",
        "model": {
            "type": model_name,
            "multilingual": "en" not in model_name
        },
        "params": {
            "model": model_name,
            "language": detected_lang,
            "translate": args.translate
        },
        "result": {
            "language": detected_lang
        },
        "transcription": transcription_list
    }

    out_base = args.output_file
    if out_base:
        out_json = f"{out_base}.json"
    else:
        out_json = "transcript.json"

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

def format_whisper_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms >= 1000: ms = 999
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

if __name__ == "__main__":
    main()
PYEOF
chmod +x /usr/local/bin/whisper-cli-gpu
ln -sf /usr/local/bin/whisper-cli-gpu /usr/local/bin/whisper-cli
mkdir -p /root/.ossclip/bin /root/.ossclip/models
ln -sf /usr/local/bin/whisper-cli-gpu /root/.ossclip/bin/whisper-cli
touch /root/.ossclip/models/ggml-small.en.bin

# 5. Create ossclip-gpu-render tool
cat << 'PYEOF' > /usr/local/bin/ossclip-gpu-render
#!/usr/bin/env python3
import sys, os, json, argparse, subprocess, re

STYLE_PRESETS = {
    "hormozi": {
        "accent": "#FFE600",
        "fontDisplay": "Montserrat",
        "fontMono": "JetBrains Mono",
        "outlinePx": 5,
        "shadowPx": 6,
        "fontSize": 48
    },
    "mrbeast": {
        "accent": "#00FFA3",
        "fontDisplay": "Montserrat",
        "fontMono": "JetBrains Mono",
        "outlinePx": 6,
        "shadowPx": 8,
        "fontSize": 52
    },
    "cyber": {
        "accent": "#00F0FF",
        "fontDisplay": "Montserrat",
        "fontMono": "JetBrains Mono",
        "outlinePx": 3,
        "shadowPx": 0,
        "fontSize": 44
    },
    "pill": {
        "accent": "#FDE047",
        "fontDisplay": "Montserrat",
        "fontMono": "JetBrains Mono",
        "outlinePx": 0,
        "shadowPx": 4,
        "fontSize": 40
    },
    "clean": {
        "accent": "#38BDF8",
        "fontDisplay": "Montserrat",
        "fontMono": "JetBrains Mono",
        "outlinePx": 2,
        "shadowPx": 2,
        "fontSize": 42
    }
}

def hex_to_ass_color(hex_str):
    if not hex_str: return "&H00FFFFFF&"
    hex_str = hex_str.lstrip('#')
    if len(hex_str) == 6:
        r, g, b = hex_str[0:2], hex_str[2:4], hex_str[4:6]
        return f"&H00{b}{g}{r}&"
    elif len(hex_str) == 8:
        a, r, g, b = hex_str[0:2], hex_str[2:4], hex_str[4:6], hex_str[6:8]
        return f"&H{a}{b}{g}{r}&"
    return "&H00FFFFFF&"

def format_ass_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int(round((seconds - int(seconds)) * 100))
    if cs >= 100: cs = 99
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

def esc(text):
    if text is None: return ""
    return str(text).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

def subtract_user_cuts(spans, user_cuts):
    """Applies cuts made by user in the Web Interface (overrides.cuts)."""
    current_spans = [(s["srcIn"], s["srcOut"]) for s in spans]
    for cut in user_cuts:
        src = cut.get("src")
        if not src: continue
        c_in, c_out = src.get("startSec", 0), src.get("endSec", 0)
        new_spans = []
        for s_in, s_out in current_spans:
            if c_out <= s_in or c_in >= s_out:
                new_spans.append((s_in, s_out))
            else:
                if c_in > s_in: new_spans.append((s_in, c_in))
                if c_out < s_out: new_spans.append((c_out, s_out))
        current_spans = new_spans
    return [{"srcIn": s[0], "srcOut": s[1]} for s in current_spans]

def generate_scene_component_svg(component, props, layout="lower-third", theme=None, dx=0, dy=0, scale=1.0, w=1920, h=1080):
    """Renders the EXACT component displayed in the OSSClip Web Editor with full layout & text edits."""
    if not theme: theme = {}
    accent = theme.get("accent", "#FFE600")
    fg = theme.get("fg", "#FFFFFF")
    card_bg = theme.get("cardBg", "#0F172A")
    card_border = theme.get("cardBorder", "#334155")
    font_display = theme.get("fontDisplay", "Montserrat")
    font_mono = theme.get("fontMono", "JetBrains Mono")

    # Placement based on Web Editor layout mode
    if layout == "split-left":
        box_x, box_y, box_w, box_h = 100, 260, 800, 560
    elif layout == "split-right":
        box_x, box_y, box_w, box_h = 1020, 260, 800, 560
    elif layout in ["blurred-behind", "full-bleed"]:
        box_x, box_y, box_w, box_h = 240, 620, 1440, 380
    else: # lower-third
        box_x, box_y, box_w, box_h = 240, 660, 1440, 340

    transform_attr = f'transform="translate({dx}, {dy}) scale({scale})"' if (dx != 0 or dy != 0 or scale != 1.0) else ''
    content_svg = ""

    if component == "TitleCard":
        eyebrow = esc(props.get("eyebrow", ""))
        title = esc(props.get("title", ""))
        sub = esc(props.get("sub", ""))
        emphasis = esc(props.get("emphasis", ""))

        items = []
        curr_y = box_y + 80
        if eyebrow:
            items.append(f'<text x="{box_x + box_w//2}" y="{curr_y}" font-family="{font_display}" font-size="28" font-weight="700" fill="#94A3B8" text-anchor="middle" letter-spacing="4">{eyebrow.upper()}</text>')
            curr_y += 75
        if emphasis:
            items.append(f'<text x="{box_x + box_w//2}" y="{curr_y}" font-family="{font_display}" font-size="110" font-weight="900" fill="{accent}" text-anchor="middle">{emphasis}</text>')
            curr_y += 100
        if title:
            items.append(f'<text x="{box_x + box_w//2}" y="{curr_y}" font-family="{font_display}" font-size="64" font-weight="900" fill="{fg}" text-anchor="middle">{title.upper()}</text>')
            curr_y += 65
        if sub and sub.lower() != title.lower():
            items.append(f'<text x="{box_x + box_w//2}" y="{curr_y}" font-family="{font_display}" font-size="30" font-weight="600" fill="#CBD5E1" text-anchor="middle">{sub}</text>')
        content_svg = "\n".join(items)

    elif component == "FlowDiagram":
        nodes = props.get("nodes", [])
        emphasize_last = props.get("emphasizeLast", True)
        n_count = len(nodes)
        chip_w = min(360, (box_w - (n_count * 80)) // max(1, n_count))
        total_w = n_count * chip_w + (n_count - 1) * 70
        start_x = box_x + (box_w - total_w) // 2
        chip_y = box_y + box_h // 2 - 45

        items = [
            f'<text x="{box_x + 60}" y="{box_y + 65}" font-family="{font_display}" font-size="22" font-weight="700" fill="#94A3B8" letter-spacing="2">ARCHITECTURE FLOW</text>'
        ]
        for i, node in enumerate(nodes):
            cx = start_x + i * (chip_w + 70)
            is_emph = (i == n_count - 1 and emphasize_last)
            bg = accent if is_emph else "#1E293B"
            text_color = "#0F172A" if is_emph else "#FFFFFF"
            border_color = accent if is_emph else "#475569"

            items.append(f'<rect x="{cx}" y="{chip_y}" width="{chip_w}" height="90" rx="16" fill="{bg}" stroke="{border_color}" stroke-width="2"/>')
            items.append(f'<text x="{cx + chip_w//2}" y="{chip_y + 55}" font-family="{font_display}" font-size="22" font-weight="800" fill="{text_color}" text-anchor="middle">{esc(node).upper()}</text>')

            if i < n_count - 1:
                items.append(f'<text x="{cx + chip_w + 35}" y="{chip_y + 55}" font-family="{font_display}" font-size="36" font-weight="900" fill="#64748B" text-anchor="middle">→</text>')
        content_svg = "\n".join(items)

    elif component == "StatCard":
        label = esc(props.get("label", "METRIC"))
        value = esc(props.get("value", "0"))
        caption = esc(props.get("caption", ""))

        items = [
            f'<text x="{box_x + 80}" y="{box_y + 130}" font-family="{font_display}" font-size="44" font-weight="800" fill="#94A3B8" letter-spacing="2">{label.upper()}</text>',
            f'<text x="{box_x + box_w - 80}" y="{box_y + 190}" font-family="{font_display}" font-size="110" font-weight="900" fill="{accent}" text-anchor="end">{value}</text>'
        ]
        if caption:
            items.append(f'<text x="{box_x + 80}" y="{box_y + 240}" font-family="{font_display}" font-size="28" font-weight="600" fill="#CBD5E1">{caption}</text>')
        content_svg = "\n".join(items)

    elif component == "StrikethroughReveal":
        lines = props.get("lines", [])
        items = [
            f'<text x="{box_x + 60}" y="{box_y + 65}" font-family="{font_display}" font-size="22" font-weight="700" fill="#94A3B8" letter-spacing="2">COMPARISON &amp; DECISION</text>'
        ]
        curr_y = box_y + 140
        for i, l in enumerate(lines):
            text = esc(l.get("text", ""))
            struck = l.get("struck", False)
            mark = l.get("mark", "none")

            mark_svg = ""
            if mark == "cross":
                mark_svg = f'<text x="{box_x + 80}" y="{curr_y}" font-family="{font_display}" font-size="38" font-weight="900" fill="#EF4444">✗</text>'
            elif mark == "check":
                mark_svg = f'<text x="{box_x + 80}" y="{curr_y}" font-family="{font_display}" font-size="38" font-weight="900" fill="#10B981">✓</text>'

            text_color = "#64748B" if struck else "#FFFFFF"
            items.append(mark_svg)
            items.append(f'<text x="{box_x + 130}" y="{curr_y}" font-family="{font_display}" font-size="38" font-weight="800" fill="{text_color}">{text.upper()}</text>')

            if struck:
                approx_w = len(text) * 24
                items.append(f'<line x1="{box_x + 125}" y1="{curr_y - 12}" x2="{box_x + 135 + approx_w}" y2="{curr_y - 12}" stroke="#EF4444" stroke-width="4"/>')
            curr_y += 75
        content_svg = "\n".join(items)

    elif component in ["ScreenshotFrame", "BrowserFrame"]:
        label = esc(props.get("label", "DASHBOARD"))
        items = [
            f'<rect x="{box_x + 40}" y="{box_y + 40}" width="{box_w - 80}" height="44" rx="8" fill="#1E293B"/>',
            f'<circle cx="{box_x + 70}" cy="{box_y + 62}" r="6" fill="#EF4444"/>',
            f'<circle cx="{box_x + 90}" cy="{box_y + 62}" r="6" fill="#F59E0B"/>',
            f'<circle cx="{box_x + 110}" cy="{box_y + 62}" r="6" fill="#10B981"/>',
            f'<text x="{box_x + 140}" y="{box_y + 69}" font-family="{font_mono}" font-size="16" fill="#94A3B8">{label.upper()}</text>',
            f'<rect x="{box_x + 40}" y="{box_y + 100}" width="{box_w - 80}" height="{box_h - 140}" rx="12" fill="#0B132B" stroke="#1E293B" stroke-width="2"/>',
            f'<text x="{box_x + box_w//2}" y="{box_y + box_h//2}" font-family="{font_display}" font-size="36" font-weight="800" fill="{accent}" text-anchor="middle">LIVE DEMO • {label}</text>'
        ]
        content_svg = "\n".join(items)

    elif component in ["TerminalMock", "terminal", "tokyo-night"]:
        title = esc(props.get("title", "convex/files.ts"))
        lines = props.get("lines", [
            'import { mutation } from "./_generated/server";',
            'export const generateUploadUrl = mutation({',
            '  handler: async (ctx) => await ctx.storage.generateUploadUrl()',
            '});'
        ])
        items = [
            f'<rect x="{box_x}" y="{box_y}" width="{box_w}" height="48" rx="20" fill="#1E293B"/>',
            f'<circle cx="{box_x + 30}" cy="{box_y + 24}" r="6" fill="#EF4444"/>',
            f'<circle cx="{box_x + 50}" cy="{box_y + 24}" r="6" fill="#F59E0B"/>',
            f'<circle cx="{box_x + 70}" cy="{box_y + 24}" r="6" fill="#10B981"/>',
            f'<text x="{box_x + 100}" y="{box_y + 30}" font-family="{font_mono}" font-size="15" fill="#94A3B8">{title}</text>'
        ]
        for i, l in enumerate(lines[:7]):
            line_no = f"{i+1:02d}"
            escaped = esc(l)
            items.append(f'<text x="{box_x + 40}" y="{box_y + 90 + i*36}" font-family="{font_mono}" font-size="18" fill="#475569">{line_no}</text>')
            items.append(f'<text x="{box_x + 85}" y="{box_y + 90 + i*36}" font-family="{font_mono}" font-size="18" fill="#C0CAF5">{escaped}</text>')
        content_svg = "\n".join(items)

    elif component == "BulletList":
        heading = esc(props.get("heading", "KEY POINTS"))
        items_list = props.get("items", [])
        items = [
            f'<text x="{box_x + 60}" y="{box_y + 70}" font-family="{font_display}" font-size="32" font-weight="900" fill="{accent}">{heading.upper()}</text>'
        ]
        for i, item in enumerate(items_list[:4]):
            items.append(f'<circle cx="{box_x + 80}" cy="{box_y + 130 + i*55}" r="7" fill="{accent}"/>')
            items.append(f'<text x="{box_x + 110}" y="{box_y + 137 + i*55}" font-family="{font_display}" font-size="28" font-weight="700" fill="#FFFFFF">{esc(item)}</text>')
        content_svg = "\n".join(items)

    else:
        title = esc(props.get("title", component))
        content_svg = f'<text x="{box_x + box_w//2}" y="{box_y + box_h//2}" font-family="{font_display}" font-size="44" font-weight="900" fill="{fg}" text-anchor="middle">{title}</text>'

    return f"""<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <filter id="shadow" x="-10%" y="-10%" width="120%" height="130%">
      <feDropShadow dx="0" dy="20" stdDeviation="30" flood-color="#000000" flood-opacity="0.85"/>
    </filter>
  </defs>
  <g {transform_attr} filter="url(#shadow)">
    <rect x="{box_x}" y="{box_y}" width="{box_w}" height="{box_h}" rx="24" fill="{card_bg}" stroke="{card_border}" stroke-width="2"/>
    {content_svg}
  </g>
</svg>"""

def render_cairo_png_from_svg(svg_str, out_png, w=1920, h=1080):
    try:
        import cairosvg
        cairosvg.svg2png(bytestring=svg_str.encode('utf-8'), write_to=out_png, output_width=w, output_height=h)
        return True
    except Exception as e:
        print(f"Warning: Cairo rendering failed: {e}")
        return False

def build_ass_subtitles(caption_lines, theme, out_ass_path, res_x=1920, res_y=1080, style_info=None):
    if not style_info: style_info = {}
    font_display = theme.get("fontDisplay", style_info.get("fontDisplay", "Montserrat"))
    font_display = font_display.replace("'", "").replace('"', '').split(',')[0].strip()
    accent_hex = theme.get("accent", style_info.get("accent", "#FFE600"))
    accent_ass = hex_to_ass_color(accent_hex)
    normal_ass = "&H00FFFFFF&"
    font_size = style_info.get("fontSize", 46)
    outline_px = style_info.get("outlinePx", 4)
    shadow_px = style_info.get("shadowPx", 4)

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {res_x}
PlayResY: {res_y}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Normal,{font_display},{font_size},{normal_ass},&H000000FF&,&H00000000&,&H80000000&,-1,0,0,0,100,100,0,0,1,{outline_px},{shadow_px},2,30,30,80,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    sub_y = int(res_y * 0.85)
    pos_tag = f"{{\\pos({res_x // 2},{sub_y})}}"

    for line in caption_lines:
        words = line.get("words", [])
        if not words: continue
        for i, current_word in enumerate(words):
            start_str = format_ass_time(current_word["start"])
            end_str = format_ass_time(current_word["end"])
            parts = []
            for j, w in enumerate(words):
                txt = w["text"].upper()
                if i == j:
                    parts.append(r"{\c" + accent_ass + r"\fscx108\fscy108}" + txt + r"{\r}")
                else:
                    parts.append(r"{\c" + normal_ass + r"}" + txt)
            events.append(f"Dialogue: 1,{start_str},{end_str},Normal,,0,0,0,,{pos_tag}{' '.join(parts)}")

    with open(out_ass_path, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(events) + "\n")

def main():
    parser = argparse.ArgumentParser(description="OSSClip GPU Exporter with Multi-Graphic Cairo & Web Editor Sync")
    parser.add_argument("workdir")
    parser.add_argument("--out", "-o", default=None)
    parser.add_argument("--format", choices=["auto", "vertical", "original", "blur-backdrop"], default="auto")
    parser.add_argument("--style", choices=list(STYLE_PRESETS.keys()) + ["default"], default="hormozi")
    parser.add_argument("--graphics-style", default="auto")
    parser.add_argument("--bitrate", default="6M")
    parser.add_argument("--no-graphics", action="store_true", help="Do not render graphic overlays")
    parser.add_argument("--no-captions", action="store_true", help="Do not burn in subtitle captions")
    args = parser.parse_args()

    workdir = args.workdir
    if os.path.isfile(workdir):
        video_path = os.path.abspath(workdir)
        parent, base = os.path.dirname(video_path), os.path.splitext(os.path.basename(video_path))[0]
        ossclip_dir = os.path.join(parent, ".ossclip")
        matched = [os.path.join(ossclip_dir, d) for d in os.listdir(ossclip_dir) if d.startswith(base)] if os.path.exists(ossclip_dir) else []
        if not matched:
            subprocess.run(["ossclip", "transcribe", video_path], check=True)
            matched = [os.path.join(ossclip_dir, d) for d in os.listdir(ossclip_dir) if d.startswith(base)]
        workdir = matched[0]

    render_props = json.load(open(os.path.join(workdir, "render-props.json"), "r", encoding="utf-8"))
    overrides = {}
    if os.path.exists(os.path.join(workdir, "overrides.json")):
        try: overrides = json.load(open(os.path.join(workdir, "overrides.json"), "r"))
        except: pass

    # Use /dev/shm (Shared RAM) for intermediate vector assets to eliminate disk I/O
    shm_dir = "/dev/shm" if os.path.exists("/dev/shm") else workdir

    settings = render_props.get("settings", {})
    prop_w = settings.get("width", 1920)
    prop_h = settings.get("height", 1080)

    if args.format == "auto":
        target_format = "vertical" if prop_h > prop_w else "original"
    else:
        target_format = args.format

    is_vertical = target_format in ["vertical", "blur-backdrop"]
    res_x = 1080 if is_vertical else prop_w
    res_y = 1920 if is_vertical else prop_h

    # Merge Theme: Base -> Preset -> Overrides from Web Editor
    style_info = STYLE_PRESETS.get(args.style, {})
    theme = {**render_props.get("theme", {}), **style_info, **overrides.get("theme", {})}

    if overrides.get("captionsHidden") is True:
        args.no_captions = True

    # 1. Apply Web Editor Retypes & Word Hides to Caption Lines
    caption_lines = render_props.get("captionLines", [])
    caption_edits = overrides.get("captions", {})
    hidden_words = overrides.get("captionWordsHidden", {})

    for line in caption_lines:
        kept_words = []
        for w in line.get("words", []):
            src_val = w.get("srcStart", w.get("start"))
            key = f"{src_val:.3f}" if src_val is not None else None
            if key and key in hidden_words:
                continue
            if key and key in caption_edits:
                w["text"] = caption_edits[key].get("text", w["text"])
            kept_words.append(w)
        line["words"] = kept_words

    # 2. Apply Web Editor Cuts to Spans
    spans = render_props.get("spans", [])
    user_cuts = overrides.get("cuts", [])
    if user_cuts:
        spans = subtract_user_cuts(spans, user_cuts)

    source_video = None
    if os.path.exists(os.path.join(workdir, "production.json")):
        source_video = json.load(open(os.path.join(workdir, "production.json"))).get("source", {}).get("path")
    if not source_video or not os.path.exists(source_video):
        for c in ["mezzanine.mp4", "mezzanine-content.mp4"]:
            p = os.path.join(workdir, c)
            if os.path.exists(p) and os.path.getsize(p) > 1000:
                source_video = p; break

    # 3. IDENTICAL GRAPHICS: Render the EXACT sceneCues displayed in the Web Editor
    graphic_overlays = []
    if not args.no_graphics and args.graphics_style != "none":
        scene_cues = render_props.get("sceneCues", [])
        scene_overrides = overrides.get("scenes", {})
        pid = os.getpid()

        for idx, cue in enumerate(scene_cues):
            comp = cue.get("component")
            if not comp or comp == "None" or cue.get("kind") == "plain":
                continue

            cue_id = cue.get("id", f"scene-{idx}")
            sc_override = scene_overrides.get(cue_id, {}) or scene_overrides.get(f"scene-{idx}", {})

            # Skip if user hid this graphic in the editor
            if sc_override.get("hidden") is True:
                continue

            # Merge all user text edits (props)
            props = {**cue.get("props", {}), **sc_override.get("props", {})}

            # Extract user positioning & scale edits (dx, dy, scale)
            elem_transforms = sc_override.get("elements", {})
            card_transform = elem_transforms.get("card", {}) or elem_transforms.get("root", {})
            dx = card_transform.get("dx", 0)
            dy = card_transform.get("dy", 0)
            scale = card_transform.get("scale", 1.0)

            # Timing: honors manual timeline nudges / pins
            start_t = cue.get("startSec", 0)
            end_t = cue.get("endSec", start_t + 5)
            if sc_override.get("timing"):
                timing = sc_override["timing"]
                start_t = timing.get("srcStart", timing.get("startSec", start_t))
                end_t = timing.get("srcEnd", timing.get("endSec", end_t))

            layout = sc_override.get("layout", cue.get("layout", "lower-third"))

            svg = generate_scene_component_svg(
                component=comp,
                props=props,
                layout=layout,
                theme=theme,
                dx=dx,
                dy=dy,
                scale=scale,
                w=res_x,
                h=res_y
            )

            png_path = os.path.join(shm_dir, f"editor_overlay_{idx}_{pid}.png")
            if render_cairo_png_from_svg(svg, png_path, w=res_x, h=res_y):
                graphic_overlays.append({
                    "path": png_path,
                    "start": start_t,
                    "end": end_t,
                    "comp": comp
                })

    ass_path = os.path.join(shm_dir, f"subtitles_custom_{os.getpid()}.ass")
    if not args.no_captions:
        build_ass_subtitles(caption_lines, theme, ass_path, res_x=res_x, res_y=res_y, style_info=style_info)

    span_conds = [f"between(t,{s['srcIn']:.3f},{s['srcOut']:.3f})" for s in spans if s.get("srcOut", 0) > s.get("srcIn", 0)]
    select_filter = "+".join(span_conds) if span_conds else "1"

    out_file = args.out
    if not out_file:
        if os.path.exists(os.path.join(workdir, "command.json")):
            try: out_file = json.load(open(os.path.join(workdir, "command.json"))).get("out")
            except: pass
    if not out_file:
        out_file = "/content/rendered_output.mp4"
    out_file = os.path.abspath(out_file)

    inputs = ["-hwaccel", "cuda", "-i", source_video]
    for g in graphic_overlays:
        inputs += ["-i", g["path"]]

    base_vf = []
    if select_filter != "1":
        base_vf.append(f"select='{select_filter}',setpts=N/FRAME_RATE/TB")

    ass_filter = f",ass={ass_path}" if not args.no_captions else ""
    if target_format == "vertical":
        base_vf.append(f"crop=ih*9/16:ih:(iw-ih*9/16)/2:0,scale=1080:1920")
    else:
        base_vf.append(f"scale={res_x}:{res_y}")

    if graphic_overlays:
        filter_parts = [f"[0:v]{','.join(base_vf)}[v0]"]
        for i, g in enumerate(graphic_overlays):
            in_v = f"v{i}"
            next_v = f"v{i+1}" if i < len(graphic_overlays) - 1 else "vlast"
            filter_parts.append(f"[{in_v}][{i+1}:v]overlay=enable='between(t,{g['start']},{g['end']})':format=auto[{next_v}]")
        filter_parts.append(f"[vlast]null{ass_filter}[outv]")
        cmd_filter = ["-filter_complex", "; ".join(filter_parts), "-map", "[outv]", "-map", "0:a"]
    else:
        filter_str = f"{','.join(base_vf)}{ass_filter}"
        cmd_filter = ["-vf", filter_str]

    if select_filter != "1":
        cmd_filter += ["-af", f"aselect='{select_filter}',asetpts=N/SR/TB"]

    gpu_nvenc_flags = [
        "-threads", "4",
        "-filter_threads", "2",
        "-c:v", "h264_nvenc",
        "-preset", "p4",
        "-surfaces", "32",
        "-rc-lookahead", "16",
        "-tune", "hq",
        "-b:v", args.bitrate,
        "-c:a", "aac", "-b:a", "192k",
        out_file
    ]

    cmd = ["ffmpeg", "-y"] + inputs + cmd_filter + gpu_nvenc_flags

    print(f"🎬 Starting GPU Export (Tesla T4 NVENC)...")
    if graphic_overlays:
        print(f"✨ Rendering EXACT Web Editor Components ({len(graphic_overlays)} graphics):")
        for idx, g in enumerate(graphic_overlays):
            print(f"   [{idx+1}] {g['comp']} at {g['start']:.2f}s -> {g['end']:.2f}s")
    else:
        print(f"⚡ Clean Cut Video (No Graphics)")

    subprocess.run(cmd, check=True)
    print(f"✅ Finished! Video written to: {out_file}")

if __name__ == "__main__":
    main()
PYEOF
chmod +x /usr/local/bin/ossclip-gpu-render

# 6. Apply GPU patches to ossclip source
python3 - << 'PYEOF'
import os

def replace_in_file(path, old, new):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        if old in content:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content.replace(old, new))

# Patch render-options.ts
replace_in_file(
    "/tools/node/lib/node_modules/ossclip/node_modules/@ossclip/renderer/src/render-options.ts",
    'hardwareAcceleration: platform === "darwin" ? "if-possible" : "disable",',
    'hardwareAcceleration: "if-possible",\n    chromiumOptions: { enableMultiProcessOnLinux: true, gl: "angle" },'
)

# Patch ingest.ts
replace_in_file(
    "/tools/node/lib/node_modules/ossclip/node_modules/@ossclip/core/src/ingest.ts",
    '"-c:v", "libx264", "-preset", "veryfast", "-crf", "18",',
    '"-c:v", "h264_nvenc", "-preset", "p4", "-cq", "18",'
)

# Web Editor uses official default Remotion engine with full layout & graphic fidelity

# Patch phonetics.ts to allow labial onsets (e.g. "parcel" -> "Vercel")
replace_in_file(
    "/tools/node/lib/node_modules/ossclip/node_modules/@ossclip/core/src/phonetics.ts",
    'if (ka[0] !== kb[0]) return false;',
    '''const onsetA = ka[0];
  const onsetB = kb[0];
  const LABIALS = new Set(["p", "b", "f", "v", "w"]);
  const isLabialMatch = LABIALS.has(onsetA) && LABIALS.has(onsetB);
  if (onsetA !== onsetB && !isLabialMatch) return false;'''
)

# Patch antigravity.ts for headless permissions and fast low-latency models (~14s vs 90s)
replace_in_file(
    "/tools/node/lib/node_modules/ossclip/node_modules/@ossclip/core/src/producer/antigravity.ts",
    '    "-p",\n    prompt,\n    "--output-format",',
    '    "-p",\n    prompt,\n    "--dangerously-skip-permissions",\n    "--output-format",'
)
replace_in_file(
    "/tools/node/lib/node_modules/ossclip/node_modules/@ossclip/core/src/producer/antigravity.ts",
    '    ...(opts.model ? ["--model", opts.model] : []),\n    // Omitted entirely when unset — agy\'s own default stands, exactly as it\n    // did before the knob existed (§143).\n    ...(opts.effort ? ["--effort", opts.effort] : []),',
    '    "--model",\n    opts.model ?? "gemini-3.7-flash-low",\n    "--effort",\n    opts.effort ?? "low",'
)

# Patch analyze.ts to prevent background room tone from vetoing silence cuts
replace_in_file(
    "/tools/node/lib/node_modules/ossclip/node_modules/@ossclip/core/src/analyze.ts",
    '      if (meanDb <= levels.speechDb - DEAD_AIR_DROP) conflicts = [];',
    '      const threshold = (levels as any).thresholdDb ?? (levels.speechDb - DEAD_AIR_DROP);\n      if (meanDb <= threshold) conflicts = [];'
)

# Patch produce.ts so AI transcript review runs even when graphics/title cards are turned off
replace_in_file(
    "/tools/node/lib/node_modules/ossclip/src/produce.ts",
    'const needsLlm = opts.produce === true || resolveYoutube(opts.youtube, cfg.youtube);',
    'const needsLlm = opts.produce === true || resolveYoutube(opts.youtube, cfg.youtube) || opts.repair !== false;'
)

# Patch produce-wizard.ts so AI graphics prompt is explicit, defaults to No, and offers Cairo styles
replace_in_file(
    "/tools/node/lib/node_modules/ossclip/src/interactive/produce-wizard.ts",
    'message: "Plan title cards and graphics with an LLM?", initialValue: false',
    'message: "Create graphics with AI? (No = clean cut video without graphic overlays)", initialValue: false'
)
replace_in_file(
    "/tools/node/lib/node_modules/ossclip/src/interactive/produce-wizard.ts",
    'initialValue: false,\n    }),\n  ) as boolean;',
    'initialValue: false,\n    }),\n  ) as boolean;\n\n  const graphicsStyle = graphics ? (unwrap(await select({\n    message: "Select Cairo AI graphics style:",\n    options: [\n      { value: "tokyo-night", label: "Tokyo Night Terminal", hint: "macOS Code & CLI Window" },\n      { value: "linear", label: "Linear Obsidian", hint: "Cloud & Data Architecture Flow" },\n      { value: "stripe", label: "Stripe Clean Paper", hint: "High-Contrast Comparison Matrix" },\n      { value: "vercel", label: "Vercel Geist Dark", hint: "Performance Metrics & KPI Cards" },\n      { value: "auto", label: "Auto Match", hint: "AI chooses based on speech topic" }\n    ]\n  })) as string) : undefined;'
)
replace_in_file(
    "/tools/node/lib/node_modules/ossclip/src/interactive/produce-wizard.ts",
    'return produceArgv({\n    input,\n    aspect,\n    cleanup,\n    graphics,',
    'return produceArgv({\n    input,\n    aspect,\n    cleanup,\n    graphics,\n    graphicsStyle,'
)
replace_in_file(
    "/tools/node/lib/node_modules/ossclip/src/interactive/produce-argv.ts",
    'graphics: boolean;',
    'graphics: boolean;\n  graphicsStyle?: string;'
)
replace_in_file(
    "/tools/node/lib/node_modules/ossclip/src/interactive/produce-argv.ts",
    'if (a.graphics) {\n    argv.push("--produce");',
    'if (a.graphics) {\n    argv.push("--produce");\n    if (a.graphicsStyle && a.graphicsStyle !== "auto") argv.push("--graphics-style", a.graphicsStyle);'
)
replace_in_file(
    "/tools/node/lib/node_modules/ossclip/src/program.ts",
    '.option("--produce", "run the LLM producer brain to plan title cards & graphics", false)',
    '.option("--produce", "run the LLM producer brain to plan title cards & graphics", false)\n    .option("--graphics-style <style>", "Cairo AI graphics style: tokyo-night | linear | stripe | vercel | auto", "auto")'
)
PYEOF

# Create default developer vocabulary dictionary & fast LLM configuration
mkdir -p /root/.ossclip
cat << 'JSONEOF' > /root/.ossclip/config.json
{
  "fastModel": "gemini-3.7-flash-low",
  "llmEffort": "low",
  "dictionary": [
    "Vercel",
    "Convex",
    "Next.js",
    "TypeScript",
    "JavaScript",
    "React",
    "API",
    "URL",
    "JSON",
    "Node.js",
    "Tailwind",
    "PostgreSQL",
    "GraphQL",
    "Prisma",
    "Docker"
  ]
}
JSONEOF

echo "✅ All OSSClip GPU Optimizations Configured Successfully!"

# 7. Install Modern Creator Fonts (Montserrat, Bebas Neue, Anton, Rubik)
mkdir -p /usr/share/fonts/truetype/custom
FONT_SOURCE_DIR=""
for d in "$SCRIPT_DIR/fonts" "/content/OSSClip_Setup/fonts" "/content/fonts" "/content/drive/MyDrive/OSSClip_Setup/fonts"; do
  if [ -d "$d" ] && [ -f "$d/Montserrat.ttf" ]; then
    FONT_SOURCE_DIR="$d"
    break
  fi
done

if [ -n "$FONT_SOURCE_DIR" ]; then
  echo "📦 Loading Creator Fonts from local directory ($FONT_SOURCE_DIR)..."
  cp "$FONT_SOURCE_DIR"/*.ttf /usr/share/fonts/truetype/custom/
else
  echo "📦 Downloading Creator Fonts from Google Fonts..."
  curl -fsSL "https://github.com/google/fonts/raw/main/ofl/montserrat/Montserrat%5Bwght%5D.ttf" -o /usr/share/fonts/truetype/custom/Montserrat.ttf
  curl -fsSL "https://github.com/google/fonts/raw/main/ofl/jetbrainsmono/JetBrainsMono%5Bwght%5D.ttf" -o /usr/share/fonts/truetype/custom/JetBrainsMono.ttf
  curl -fsSL "https://github.com/google/fonts/raw/main/ofl/bebasneue/BebasNeue-Regular.ttf" -o /usr/share/fonts/truetype/custom/BebasNeue.ttf
  curl -fsSL "https://github.com/google/fonts/raw/main/ofl/anton/Anton-Regular.ttf" -o /usr/share/fonts/truetype/custom/Anton.ttf
  curl -fsSL "https://github.com/google/fonts/raw/main/ofl/rubik/Rubik%5Bwght%5D.ttf" -o /usr/share/fonts/truetype/custom/Rubik.ttf
fi
fc-cache -f -v > /dev/null

# 8. Restore Antigravity Video Editing Skill
echo "🧠 Restoring Antigravity Video Editing Skill..."
mkdir -p /content/.agents/skills/ossclip-gpu-video-editing \
         /root/.gemini/config/skills/ossclip-gpu-video-editing \
         /root/.gemini/antigravity-cli/skills/ossclip-gpu-video-editing \
         /root/.agents/skills/ossclip-gpu-video-editing

SKILL_SOURCE=""
for s in "$SCRIPT_DIR/.agents/skills/ossclip-gpu-video-editing/SKILL.md" \
         "/content/OSSClip_Setup/.agents/skills/ossclip-gpu-video-editing/SKILL.md" \
         "/content/.agents/skills/ossclip-gpu-video-editing/SKILL.md" \
         "/content/drive/MyDrive/OSSClip_Setup/.agents/skills/ossclip-gpu-video-editing/SKILL.md"; do
  if [ -f "$s" ]; then
    SKILL_SOURCE="$s"
    break
  fi
done

if [ -n "$SKILL_SOURCE" ]; then
  echo "📦 Restoring skill from $SKILL_SOURCE..."
  for dest in "/content/.agents/skills/ossclip-gpu-video-editing/SKILL.md" \
              "/root/.gemini/config/skills/ossclip-gpu-video-editing/SKILL.md" \
              "/root/.gemini/antigravity-cli/skills/ossclip-gpu-video-editing/SKILL.md" \
              "/root/.agents/skills/ossclip-gpu-video-editing/SKILL.md"; do
    if [ "$SKILL_SOURCE" != "$dest" ]; then
      cp -f "$SKILL_SOURCE" "$dest"
    fi
  done
else
  echo "📦 Writing embedded Antigravity Video Editing Skill..."
  cat << 'SKILLEOF' | tee /content/.agents/skills/ossclip-gpu-video-editing/SKILL.md \
                         /root/.gemini/config/skills/ossclip-gpu-video-editing/SKILL.md \
                         /root/.gemini/antigravity-cli/skills/ossclip-gpu-video-editing/SKILL.md \
                         /root/.agents/skills/ossclip-gpu-video-editing/SKILL.md > /dev/null
---
name: ossclip-gpu-video-editing
description: Video editing with OSSClip, GPU Whisper ASR, AI silence removal, viral caption styling, and Tesla T4 NVENC GPU rendering. Use when transcribing, editing, cutting, styling captions, or rendering videos on Google Colab or Linux with an NVIDIA GPU.
---

# OSSClip GPU Video Editing

Runbook for processing, transcribing, editing, and rendering videos using OSSClip with Tesla T4 NVENC hardware acceleration, CUDA Whisper transcription, and viral caption styling.

---

## 1. Environment Verification & Restore

If running in a fresh Google Colab session, run the automated setup script to ensure Node.js v22, `faster-whisper`, `ossclip-gpu-render`, and creator fonts (`Montserrat`, `Bebas Neue`, `Anton`, `Rubik`) are installed:

```bash
# Run setup directly in working directory:
bash setup_ossclip_gpu.sh
# Or with explicit path:
bash /content/OSSClip_Setup/setup_ossclip_gpu.sh
```

**Verification checklist:**
- Node.js >= 22: `node -v`
- GPU Whisper CLI: `/usr/local/bin/whisper-cli --help`
- GPU Exporter: `/usr/local/bin/ossclip-gpu-render --help`
- Creator Fonts: `fc-list : family | grep -E "Montserrat|Bebas|Anton"`

---

## 2. Ingest & GPU Transcription

Transcribe video speech, detect silence/pauses, and generate cutlists on the Tesla T4 GPU:

```bash
ossclip transcribe <input_video.mp4>
```

- **Output:** Creates `.ossclip/<video_name>-<hash>/` with `transcript.json`, `render-props.json`, and `report.txt`.
- **Performance:** ~18 seconds for 1m 44s, ~37 seconds for 17 minutes on Tesla T4.

---

## 3. Interactive Web Editor

Launch the local timeline editor to review cuts, tweak words, and adjust scene framing:

```bash
# Start editor in background
ossclip edit /content/.ossclip/<workdir_name> --port 5174
```

In Google Colab, display the UI directly in a notebook cell:
```python
from google.colab.output import serve_kernel_port_as_iframe
serve_kernel_port_as_iframe(5174, height=800)
```

- Any edits to cuts, captions, fonts, or colors are saved automatically to `overrides.json`.
- Clicking the **"Render"** button in the Web UI triggers `/usr/local/bin/ossclip-gpu-render` directly at GPU speed (~25-30s).

---

## 4. Hardware-Accelerated GPU Rendering (`ossclip-gpu-render`)

Use `ossclip-gpu-render` to render final videos using Tesla T4 NVENC hardware encoding (~115+ fps) instead of the CPU browser screenshot loop.

### Command Syntax

```bash
ossclip-gpu-render <workdir_or_video_path> [options]
```

### Options

| Flag | Values | Description |
|---|---|---|
| `--format` | `vertical`, `original`, `blur-backdrop` | Aspect ratio framing (`vertical` = 9:16 vertical crop for Shorts/Reels/TikTok; `original` = 16:9 widescreen). |
| `--style` | `hormozi`, `mrbeast`, `cyber`, `pill`, `clean` | Viral caption style preset. |
| `--no-graphics` | (flag) | Disables all AI scene cards, rule cards, diagrams, and graphic overlays. |
| `--no-captions` | (flag) | Disables burned-in subtitles for clean video export. |
| `--out`, `-o` | `<path.mp4>` | Output video file path. |
| `--bitrate` | `6M`, `8M` | Video bitrate (default: `6M`). |

### Style Presets

- **`hormozi`** (Default): Bold **Montserrat ExtraBold**, Electric Yellow active-word pop (`#FFE600`), 5px black outline, soft drop shadow.
- **`mrbeast`**: **Bebas Neue** / **Anton**, Neon Green active-word pop (`#22FF77`), 6px heavy punch outline.
- **`cyber`**: **Anton**, Vibrant Cyan active-word pop (`#00F0FF`), high contrast stroke.
- **`pill`**: **Montserrat**, Gold Yellow highlight on a translucent dark rounded pill card.
- **`clean`**: **Rubik**, minimalist white typography with soft shadow.

### Examples

```bash
# 9:16 Vertical Reel with Alex Hormozi captions:
ossclip-gpu-render /content/input_video.mp4 --format vertical --style hormozi --out /content/output_reel.mp4

# 16:9 Widescreen with 161 silence cuts & captions:
ossclip-gpu-render /content/long_video.mp4 --format original --out /content/output_edited.mp4
```

---

## 5. Reference & Architecture

- **`transcript.json`**: AI word-level timestamps generated by CUDA Float16 Whisper.
- **`render-props.json`**: Kept spans (silence cuts), caption lines, and stage dimensions.
- **`overrides.json`**: User modifications from the Web Editor (font, color, cuts, scale).
- **`subtitles_custom.ass`**: Dynamically generated Advanced SubStation Alpha script with karaoke word animation tags (`\fscx115\fscy115\c&H...`).
- **`FFmpeg NVENC`**: `-c:v h264_nvenc -preset p4` compiles the video and subtitles at 115–200 fps.
SKILLEOF
fi

echo "✨ Full OSSClip GPU Video Editing Suite Ready!"


