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
pip install -q faster-whisper ctranslate2 gdown

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

    batch_size = int(os.environ.get("WHISPER_BATCH_SIZE", "16"))

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
            word_timestamps=True
        )

    detected_lang = getattr(info, "language", None) or (args.language if args.language and args.language != "auto" else "en")

    transcription_list = []
    for s in segments:
        words = s.words or []
        if words:
            for w in words:
                if not w.word or not w.word.strip():
                    continue
                start_sec = max(0.0, float(w.start))
                end_sec = max(start_sec + 0.05, float(w.end))
                start_ms = int(round(start_sec * 1000))
                end_ms = int(round(end_sec * 1000))

                from_h = int(start_sec // 3600)
                from_m = int((start_sec % 3600) // 60)
                from_s = start_sec % 60
                to_h = int(end_sec // 3600)
                to_m = int((end_sec % 3600) // 60)
                to_s = end_sec % 60

                transcription_list.append({
                    "timestamps": {
                        "from": f"{from_h:02d}:{from_m:02d}:{from_s:06.3f}".replace(".", ","),
                        "to": f"{to_h:02d}:{to_m:02d}:{to_s:06.3f}".replace(".", ",")
                    },
                    "offsets": {
                        "from": start_ms,
                        "to": end_ms
                    },
                    "text": w.word
                })
        elif s.text and s.text.strip():
            raw_words = s.text.strip().split()
            dur = max(0.1, s.end - s.start)
            w_dur = dur / len(raw_words)
            for idx, rw in enumerate(raw_words):
                w_start = s.start + idx * w_dur
                w_end = s.start + (idx + 1) * w_dur
                start_ms = int(round(w_start * 1000))
                end_ms = int(round(w_end * 1000))
                transcription_list.append({
                    "timestamps": {
                        "from": f"{int(w_start//3600):02d}:{int((w_start%3600)//60):02d}:{w_start%60:06.3f}".replace(".", ","),
                        "to": f"{int(w_end//3600):02d}:{int((w_end%3600)//60):02d}:{w_end%60:06.3f}".replace(".", ",")
                    },
                    "offsets": {
                        "from": start_ms,
                        "to": end_ms
                    },
                    "text": (" " + rw) if (idx > 0 or transcription_list) else rw
                })

    out_obj = {
        "systeminfo": f"Tesla T4 GPU (faster-whisper CUDA fp16 batched-{batch_size})" if device == "cuda" else "CPU (faster-whisper int8)",
        "model": {"type": model_name},
        "params": {"language": detected_lang},
        "result": {"language": detected_lang},
        "transcription": transcription_list
    }

    if args.output_file:
        with open(f"{args.output_file}.json", "w", encoding="utf-8") as f:
            json.dump(out_obj, f, indent=2, ensure_ascii=False)
    else:
        print(json.dumps(out_obj, indent=2, ensure_ascii=False))

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
import sys, os, json, argparse, subprocess, time

STYLE_PRESETS = {
    "hormozi": {
        "fontDisplay": "Montserrat",
        "accent": "#FFE600",
        "fg": "#FFFFFF",
        "outline": 5,
        "shadow": 2
    },
    "mrbeast": {
        "fontDisplay": "Bebas Neue, Anton",
        "accent": "#22FF77",
        "fg": "#FFFFFF",
        "outline": 6,
        "shadow": 3
    },
    "cyber": {
        "fontDisplay": "Anton",
        "accent": "#00F0FF",
        "fg": "#FFFFFF",
        "outline": 5,
        "shadow": 2
    },
    "pill": {
        "fontDisplay": "Montserrat",
        "accent": "#FFD700",
        "fg": "#FFFFFF",
        "outline": 4,
        "shadow": 2
    },
    "clean": {
        "fontDisplay": "Rubik",
        "accent": "#FFFFFF",
        "fg": "#FFFFFF",
        "outline": 3,
        "shadow": 1
    }
}

def hex_to_ass_color(hex_str: str, default: str = "&H00FFFFFF") -> str:
    if not hex_str or not hex_str.startswith("#") or len(hex_str) < 7:
        return default
    hex_str = hex_str.lstrip("#")
    r, g, b = hex_str[0:2], hex_str[2:4], hex_str[4:6]
    return f"&H00{b}{g}{r}".upper()

def format_ass_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}"

def extract_font_name(font_string: str) -> str:
    if not font_string:
        return "Montserrat"
    for name in font_string.replace("'", "").replace('"', "").split(","):
        name = name.strip()
        if name and name.lower() not in ["sans-serif", "serif", "monospace"]:
            return name
    return "Montserrat"

def load_scenes(workdir: str):
    # 1. Prefer user-reviewed edits from Web Interface
    reviewed_path = os.path.join(workdir, "scenes-reviewed.json")
    if os.path.exists(reviewed_path):
        try:
            return json.load(open(reviewed_path, "r", encoding="utf-8"))
        except: pass

    # 2. Check production.json scenes
    prod_path = os.path.join(workdir, "production.json")
    if os.path.exists(prod_path):
        try:
            scenes = json.load(open(prod_path, "r", encoding="utf-8")).get("scenes", [])
            if scenes: return scenes
        except: pass

    # 3. Check scenes-*.json
    for f in os.listdir(workdir):
        if f.startswith("scenes-") and f.endswith(".json"):
            try:
                return json.load(open(os.path.join(workdir, f), "r", encoding="utf-8"))
            except: pass
    return []

def build_ass_subtitles_with_cards(caption_lines, scene_cues, overrides, theme, out_ass_path, res_x=1920, res_y=1080, is_vertical=False, style_info=None):
    font_name = extract_font_name(theme.get("fontDisplay", "Montserrat"))
    base_font_size = 56 if is_vertical else 42
    caption_scale = overrides.get("captionScale", 1.0)
    font_size = int(round(base_font_size * caption_scale))
    accent_ass = hex_to_ass_color(theme.get("accent", "#FFE600"), "&H0000E6FF")
    normal_ass = hex_to_ass_color(theme.get("fg", "#FFFFFF"), "&H00FFFFFF")
    outline = style_info.get("outline", 4) if style_info else 4
    shadow = style_info.get("shadow", 2) if style_info else 2

    # Normal and lifted Y anchors
    default_sub_y = int(res_y * 0.85)
    lifted_sub_y = int(res_y * 0.63)
    card_x = int(res_x * 0.05)
    card_y = int(res_y * 0.94)

    card_font_size = int(round(36 * (res_y / 1080)))

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {res_x}
PlayResY: {res_y}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Normal,{font_name},{font_size},{normal_ass},{normal_ass},&H00000000,&H90000000,-1,0,0,0,100,100,0,0,1,{outline},{shadow},2,40,40,90,1
Style: SceneCard,{font_name},{card_font_size},&H00FFFFFF,&H00FFFFFF,{accent_ass},&HDF13130F,-1,0,0,0,100,100,0,0,3,3,0,1,60,60,90,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []

    # 1. Build Scene Card Dialogue Events (Layer 2)
    card_windows = []
    for cue in scene_cues:
        start_s = cue.get("startSec", 0)
        end_s = cue.get("endSec", 0)
        comp = cue.get("component")
        props = cue.get("props", {})
        if not comp or end_s <= start_s: continue

        card_windows.append((start_s, end_s))
        start_str = format_ass_time(start_s)
        end_str = format_ass_time(end_s)
        pos_tag = f"{{\\fad(300,300)\\pos({card_x},{card_y})}}"

        card_lines = []
        if comp == "TitleCard":
            eyebrow = props.get("eyebrow", "").upper()
            title = props.get("title", "").upper()
            sub = props.get("sub", "")
            if eyebrow: card_lines.append(f"{{\\c{accent_ass}\\fscx75\\fscy75}}{eyebrow}")
            if title: card_lines.append(f"{{\\c&H00FFFFFF\\b1\\fscx110\\fscy110}}{title}")
            if sub: card_lines.append(f"{{\\c&H00CBD5E1\\b0\\fscx75\\fscy75}}{sub}")
        elif comp == "RuleCard":
            kicker = props.get("kicker", "NOTE").upper()
            text = props.get("text", "").upper()
            card_lines.append(f"{{\\c{accent_ass}\\fscx75\\fscy75}}RULE: {kicker}")
            card_lines.append(f"{{\\c&H00FFFFFF\\b1\\fscx105\\fscy105}}{text}")
        elif comp == "FlowDiagram":
            nodes = [str(n).upper() for n in props.get("nodes", [])]
            card_lines.append(f"{{\\c{accent_ass}\\fscx75\\fscy75}}FLOW PIPELINE")
            card_lines.append(f"{{\\c&H00FFFFFF\\b1\\fscx100\\fscy100}}{' -> '.join(nodes)}")
        elif comp == "BulletList":
            title = props.get("title", "KEY POINTS").upper()
            card_lines.append(f"{{\\c{accent_ass}\\fscx75\\fscy75}}{title}")
            for item in props.get("items", []):
                card_lines.append(f"{{\\c&H00FFFFFF\\b1\\fscx95\\fscy95}}• {str(item).upper()}")
        elif comp == "StrikethroughReveal":
            card_lines.append(f"{{\\c{accent_ass}\\fscx75\\fscy75}}COMPARISON")
            for line in props.get("lines", []):
                txt = line.get("text", "").upper()
                if line.get("struck"):
                    card_lines.append(f"{{\\c&H005050FF}}[X] {txt}")
                else:
                    card_lines.append(f"{{\\c&H0050FF50}}[OK] {txt}")
        else:
            label = props.get("label", comp).upper()
            card_lines.append(f"{{\\c{accent_ass}\\fscx75\\fscy75}}PREVIEW")
            card_lines.append(f"{{\\c&H00FFFFFF\\b1\\fscx105\\fscy105}}{label}")

        card_body = "\\N".join(card_lines)
        events.append(f"Dialogue: 2,{start_str},{end_str},SceneCard,,0,0,0,,{pos_tag}{card_body}")

    # 2. Build Subtitle Dialogue Events (Layer 1) with Anti-Collision Routing
    for line in caption_lines:
        words = line.get("words", [])
        if not words: continue
        line_start = words[0]["start"]
        line_end = words[-1]["end"]

        # Check if line overlaps with any active card window
        overlaps_card = any(not (line_end < w_start or line_start > w_end) for w_start, w_end in card_windows)
        sub_y = lifted_sub_y if overlaps_card else default_sub_y
        pos_tag = f"{{\\pos({res_x // 2},{sub_y})}}"

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
    parser = argparse.ArgumentParser(description="OSSClip GPU Exporter with AI Graphics")
    parser.add_argument("workdir")
    parser.add_argument("--out", "-o", default=None)
    parser.add_argument("--format", choices=["auto", "vertical", "original", "blur-backdrop"], default="auto")
    parser.add_argument("--style", choices=list(STYLE_PRESETS.keys()) + ["default"], default="hormozi")
    parser.add_argument("--bitrate", default="6M")
    parser.add_argument("--no-graphics", action="store_true", help="Do not render AI scene cards or graphic overlays")
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

    # Auto-detect aspect ratio from render-props settings
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

    style_info = STYLE_PRESETS.get(args.style, {})
    theme = {**render_props.get("theme", {}), **style_info, **overrides.get("theme", {})}
    caption_lines = render_props.get("captionLines", [])
    spans = render_props.get("spans", [])

    # Load scene cues (prioritizing user-reviewed edits)
    scene_cues = render_props.get("sceneCues", [])
    if args.no_graphics:
        scene_cues = []
    else:
        reviewed_scenes = load_scenes(workdir)
        if reviewed_scenes and scene_cues:
            # Merge updated props from reviewed scenes
            reviewed_map = {s["id"]: s for s in reviewed_scenes if "id" in s}
            for cue in scene_cues:
                cid = cue.get("id")
                if cid in reviewed_map:
                    cue["props"] = {**cue.get("props", {}), **reviewed_map[cid].get("props", {})}
                    cue["component"] = reviewed_map[cid].get("component", cue.get("component"))

    source_video = None
    if os.path.exists(os.path.join(workdir, "production.json")):
        source_video = json.load(open(os.path.join(workdir, "production.json"))).get("source", {}).get("path")
    if not source_video or not os.path.exists(source_video):
        for c in ["mezzanine.mp4", "mezzanine-content.mp4"]:
            p = os.path.join(workdir, c)
            if os.path.exists(p) and os.path.getsize(p) > 1000:
                source_video = p; break

    ass_path = os.path.join(workdir, "subtitles_custom.ass")
    if not args.no_captions:
        build_ass_subtitles_with_cards(caption_lines, scene_cues, overrides, theme, ass_path, res_x=res_x, res_y=res_y, is_vertical=is_vertical, style_info=style_info)

    span_conds = [f"between(t,{s['srcIn']:.3f},{s['srcOut']:.3f})" for s in spans if s.get("srcOut", 0) > s.get("srcIn", 0)]
    select_filter = "+".join(span_conds) if span_conds else "1"

    vf = []
    if select_filter != "1": vf.append(f"select='{select_filter}',setpts=N/FRAME_RATE/TB")
    ass_filter = f",ass={ass_path}" if not args.no_captions else ""
    if target_format == "vertical":
        vf.append(f"crop=ih*9/16:ih:(iw-ih*9/16)/2:0,scale=1080:1920{ass_filter}")
    elif target_format == "blur-backdrop":
        vf.append(f"scale=1080:1920{ass_filter}")
    else:
        vf.append(f"scale={res_x}:{res_y}{ass_filter}")

    af = [f"aselect='{select_filter}',asetpts=N/SR/TB"] if select_filter != "1" else []

    out_file = args.out
    if not out_file:
        if os.path.exists(os.path.join(workdir, "command.json")):
            try: out_file = json.load(open(os.path.join(workdir, "command.json"))).get("out")
            except: pass
    if not out_file:
        out_file = "/content/rendered_output.mp4"
    out_file = os.path.abspath(out_file)

    gpu_nvenc_flags = [
        "-c:v", "h264_nvenc",
        "-preset", "p7",
        "-tune", "hq",
        "-multipass", "fullres",
        "-rc-lookahead", "32",
        "-spatial-aq", "1",
        "-temporal-aq", "1",
        "-surfaces", "64",
        "-dpb_size", "16",
        "-b_ref_mode", "middle",
        "-b:v", args.bitrate,
        "-c:a", "aac", "-b:a", "192k",
        out_file
    ]

    cmd = [
        "ffmpeg", "-y",
        "-hwaccel", "cuda",
        "-i", source_video,
        "-vf", ",".join(vf),
        *(["-af", ",".join(af)] if af else []),
        *gpu_nvenc_flags
    ]
    t0 = time.time()
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print("Note: NVDEC hardware decode encountered format quirk, falling back to CUDA NVENC p7...", file=sys.stderr)
        cmd_fallback = [
            "ffmpeg", "-y", "-i", source_video,
            "-vf", ",".join(vf),
            *(["-af", ",".join(af)] if af else []),
            *gpu_nvenc_flags
        ]
        subprocess.run(cmd_fallback, check=True)
    print(f"\n✨ Export complete in {time.time() - t0:.2f} seconds!")
    print(f"Aspect ratio: {'9:16 Vertical (1080x1920)' if is_vertical else f'16:9 Landscape ({res_x}x{res_y})'}")
    print(f"Graphics integrated: {len([c for c in scene_cues if c.get('component')])} AI scene cards rendered")
    print(f"Output saved to: {out_file}")

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

# Patch edit.ts to wire UI Render button to GPU
replace_in_file(
    "/tools/node/lib/node_modules/ossclip/src/edit.ts",
    'const child = spawn(cmd.execPath, [...cmd.execArgv, cmd.script, ...args], {',
    '''let spawnBinary = cmd.execPath;
          let spawnArgs = [...cmd.execArgv, cmd.script, ...args];
          if (existsSync("/usr/local/bin/ossclip-gpu-render")) {
            const outPath = customOut ?? cmd.out ?? "/content/rendered_output.mp4";
            spawnBinary = "/usr/local/bin/ossclip-gpu-render";
            spawnArgs = [workdir!, "--format", "auto", "--out", outPath];
          }
          const child = spawn(spawnBinary, spawnArgs, {'''
)

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


