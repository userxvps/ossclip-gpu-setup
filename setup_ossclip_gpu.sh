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
        "fontSizeLandscape": 48,
        "fontSizePortrait": 56
    },
    "mrbeast": {
        "accent": "#00FFA3",
        "fontDisplay": "Montserrat",
        "fontMono": "JetBrains Mono",
        "outlinePx": 6,
        "shadowPx": 8,
        "fontSizeLandscape": 52,
        "fontSizePortrait": 60
    },
    "cyber": {
        "accent": "#00F0FF",
        "fontDisplay": "Montserrat",
        "fontMono": "JetBrains Mono",
        "outlinePx": 3,
        "shadowPx": 0,
        "fontSizeLandscape": 44,
        "fontSizePortrait": 52
    },
    "pill": {
        "accent": "#FDE047",
        "fontDisplay": "Montserrat",
        "fontMono": "JetBrains Mono",
        "outlinePx": 0,
        "shadowPx": 4,
        "fontSizeLandscape": 40,
        "fontSizePortrait": 50
    },
    "clean": {
        "accent": "#38BDF8",
        "fontDisplay": "Montserrat",
        "fontMono": "JetBrains Mono",
        "outlinePx": 2,
        "shadowPx": 2,
        "fontSizeLandscape": 42,
        "fontSizePortrait": 52
    }
}

def normalize_layout(layout_str):
    if not layout_str: return "lower-third"
    raw = str(layout_str).strip().lower().replace("_", "-").replace(" ", "-")
    if raw in ["graphic", "graphic-only", "graphiconly", "diagram-only", "graphic only"]:
        return "graphic-only"
    if raw in ["blur", "blurred", "blur-behind", "blurred-behind", "blurbehind", "blur behind", "blurred behind"]:
        return "blurred-behind"
    if raw in ["pip", "pip-bubble", "pipbubble", "bubble", "pip bubble"]:
        return "pip-bubble"
    if raw in ["split-left", "splitleft", "left-split", "split-l", "split left"]:
        return "split-left"
    if raw in ["split-right", "splitright", "right-split", "split-r", "split right"]:
        return "split-right"
    if raw in ["video-top", "videotop", "top", "video_top", "video top"]:
        return "video-top"
    if raw in ["lower-third", "lowerthird", "lower", "lower-3rd", "lower third"]:
        return "lower-third"
    if raw in ["full-bleed", "fullbleed", "full", "full bleed"]:
        return "full-bleed"
    return raw

def normalize_component(comp_str):
    if not comp_str: return "TitleCard"
    raw = str(comp_str).strip().lower().replace("_", "-").replace(" ", "-")
    if raw in ["title", "titlecard", "title-card", "titiel", "titiel-card", "titielcard", "titiel card", "title card", "hook"]:
        return "TitleCard"
    if raw in ["flowchart", "flow-chart", "flow_chart", "flow chart", "flowdiagram", "flow-diagram", "flow_diagram", "flow diagram", "flow", "diagram", "pipeline"]:
        return "FlowDiagram"
    if raw in ["stat", "statcard", "stat-card", "stat_card", "stat card", "metric", "kpi", "number"]:
        return "StatCard"
    if raw in ["rule", "rulecard", "rule-card", "rule_card", "rule card", "best-practice", "bestpractice", "best practice", "rule"]:
        return "RuleCard"
    if raw in ["strike", "strikethrough", "strike-through", "strike_through", "strike through", "strikethroughreveal", "strikethrough-reveal", "comparison", "decision"]:
        return "StrikethroughReveal"
    if raw in ["list", "bulletlist", "bullet-list", "bullet_list", "bullet list", "bullets", "enumeration"]:
        return "BulletList"
    if raw in ["terminal", "terminalmock", "terminal-mock", "terminal_mock", "terminal mock", "code", "cli", "shell"]:
        return "TerminalMock"
    if raw in ["chat", "chatmock", "chat-mock", "chat_mock", "chat mock", "cta", "messages", "conversation"]:
        return "ChatMock"
    if raw in ["screenshot", "screenshotframe", "screenshot-frame", "screenshot_frame", "screenshot frame", "browser", "browserframe", "browser-frame", "dashboard"]:
        return "ScreenshotFrame"
    return comp_str

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

def wrap_text_to_width(text, max_w, font_size, char_w_ratio=0.62):
    if not text: return []
    words = str(text).split()
    if not words: return []
    
    char_limit = max(1, int(max_w / max(1, (font_size * char_w_ratio))))
    lines = []
    curr = []
    curr_len = 0
    
    for word in words:
        w_len = len(word)
        if not curr:
            curr.append(word)
            curr_len = w_len
        elif curr_len + 1 + w_len <= char_limit:
            curr.append(word)
            curr_len += 1 + w_len
        else:
            lines.append(" ".join(curr))
            curr = [word]
            curr_len = w_len
    if curr:
        lines.append(" ".join(curr))
    return lines

def get_stage_geometry(layout, is_landscape=True, w=1920, h=1080):
    """
    Implements Remotion stage.ts exact slot math for both landscape and portrait.
    """
    layout = normalize_layout(layout)
    if layout == "lower-third":
        g = {"x": 0.05, "y": 0.70, "w": 0.62, "h": 0.20} if is_landscape else {"x": 0.04, "y": 0.56, "w": 0.92, "h": 0.22}
        caption_anchor = 0.62 if is_landscape else 0.49
        video_mode = "full"
    elif layout == "split-left":
        if is_landscape:
            g = {"x": 0.54, "y": 0.16, "w": 0.40, "h": 0.64}
            caption_anchor = 0.84
            video_mode = "split-left"
        else:
            g = {"x": 0.04, "y": 0.54, "w": 0.92, "h": 0.24}
            caption_anchor = 0.52
            video_mode = "split-left"
    elif layout == "split-right":
        if is_landscape:
            g = {"x": 0.06, "y": 0.16, "w": 0.40, "h": 0.64}
            caption_anchor = 0.84
            video_mode = "split-right"
        else:
            g = {"x": 0.04, "y": 0.54, "w": 0.92, "h": 0.24}
            caption_anchor = 0.52
            video_mode = "split-right"
    elif layout == "video-top":
        g = {"x": 0.08, "y": 0.48, "w": 0.84, "h": 0.40} if is_landscape else {"x": 0.04, "y": 0.52, "w": 0.92, "h": 0.26}
        caption_anchor = 0.84 if is_landscape else 0.48
        video_mode = "video-top"
    elif layout == "graphic-only":
        g = {"x": 0.06, "y": 0.12, "w": 0.88, "h": 0.68} if is_landscape else {"x": 0.04, "y": 0.14, "w": 0.92, "h": 0.58}
        caption_anchor = 0.84 if is_landscape else 0.75
        video_mode = "graphic-only"
    elif layout == "pip-bubble":
        g = {"x": 0.06, "y": 0.12, "w": 0.88, "h": 0.48} if is_landscape else {"x": 0.04, "y": 0.14, "w": 0.92, "h": 0.46}
        caption_anchor = 0.78 if is_landscape else 0.65
        video_mode = "pip-bubble"
    elif layout == "blurred-behind":
        g = {"x": 0.08, "y": 0.18, "w": 0.84, "h": 0.52} if is_landscape else {"x": 0.04, "y": 0.22, "w": 0.92, "h": 0.44}
        caption_anchor = 0.82 if is_landscape else 0.69
        video_mode = "blur"
    else: # full-bleed
        g = {"x": 0.08, "y": 0.20, "w": 0.84, "h": 0.50} if is_landscape else {"x": 0.04, "y": 0.22, "w": 0.92, "h": 0.44}
        caption_anchor = 0.80 if is_landscape else 0.70
        video_mode = "full"

    return {
        "x": int(g["x"] * w),
        "y": int(g["y"] * h),
        "w": int(g["w"] * w),
        "h": int(g["h"] * h),
        "caption_anchor": caption_anchor,
        "video_mode": video_mode
    }

def generate_scene_svg(component, props, layout="lower-third", theme=None, dx=0, dy=0, scale=1.0, w=1920, h=1080):
    if not theme: theme = {}
    accent = theme.get("accent", "#FFE600")
    fg = theme.get("fg", "#FFFFFF")
    font_display = theme.get("fontDisplay", "Montserrat")
    font_mono = theme.get("fontMono", "JetBrains Mono")
    is_landscape = (w > h)

    component = normalize_component(component)
    layout = normalize_layout(layout)

    geom = get_stage_geometry(layout, is_landscape=is_landscape, w=w, h=h)
    box_x, box_y, box_w, box_h = geom["x"], geom["y"], geom["w"], geom["h"]

    transform_attr = f'transform="translate({dx}, {dy}) scale({scale})"' if (dx != 0 or dy != 0 or scale != 1.0) else ''
    content_svg = ""

    pad_x = 44 if is_landscape else 36
    pad_y = 36 if is_landscape else 30
    inner_w = box_w - (pad_x * 2)
    inner_h = box_h - (pad_y * 2)
    center_x = box_x + box_w // 2

    if component == "TitleCard":
        eyebrow = str(props.get("eyebrow", "") or "").strip()
        title = str(props.get("title", "") or "").strip()
        sub = str(props.get("sub", "") or "").strip()
        emphasis = str(props.get("emphasis", "") or "").strip()

        if emphasis and title and emphasis.lower() in title.lower():
            title = ""

        # Title Card in Lower-Third (Left Aligned on Desktop)
        if layout == "lower-third" and is_landscape:
            items = []
            eyebrow_fsize = 18
            title_fsize = 40
            sub_fsize = 20

            # Scale down if tight
            title_lines = wrap_text_to_width(title, inner_w, title_fsize, 0.65)
            if len(title_lines) > 2:
                title_fsize = 32
                title_lines = wrap_text_to_width(title, inner_w, title_fsize, 0.65)

            total_h = 0
            if eyebrow: total_h += eyebrow_fsize + 14
            total_h += len(title_lines) * (title_fsize * 1.15)
            if sub and sub.lower() != title.lower(): total_h += sub_fsize + 12

            curr_y = box_y + (box_h - total_h) // 2 + 20
            start_x = box_x + pad_x

            if eyebrow:
                items.append(f'<text x="{start_x}" y="{curr_y}" font-family="{font_display}" font-size="{eyebrow_fsize}" font-weight="700" fill="#94A3B8" letter-spacing="3">{esc(eyebrow).upper()}</text>')
                curr_y += eyebrow_fsize + 14

            for line in title_lines:
                items.append(f'<text x="{start_x}" y="{curr_y}" font-family="{font_display}" font-size="{title_fsize}" font-weight="900" fill="{fg}">{esc(line).upper()}</text>')
                curr_y += int(title_fsize * 1.15)

            if sub and sub.lower() != title.lower():
                curr_y += 4
                sub_lines = wrap_text_to_width(sub, inner_w, sub_fsize, 0.58)
                for line in sub_lines[:2]:
                    items.append(f'<text x="{start_x}" y="{curr_y}" font-family="{font_display}" font-size="{sub_fsize}" font-weight="600" fill="#CBD5E1">{esc(line)}</text>')
                    curr_y += int(sub_fsize * 1.2)
            content_svg = "\n".join(items)

        else:
            # Centered Title Card (Desktop Hero & Mobile Vertical)
            eyebrow_fsize = 20 if is_landscape else 18
            emphasis_fsize = 80 if is_landscape else 72
            title_fsize = 48 if is_landscape else 42
            sub_fsize = 24 if is_landscape else 22

            if emphasis:
                max_emp_w = inner_w * 0.85
                emp_char_w = len(emphasis) * emphasis_fsize * 0.70
                if emp_char_w > max_emp_w:
                    emphasis_fsize = int(max_emp_w / (len(emphasis) * 0.70))

            title_lines = wrap_text_to_width(title, inner_w, title_fsize, 0.65) if title else []
            if len(title_lines) > 2 or (emphasis and len(title_lines) > 1):
                title_fsize = 36 if is_landscape else 32
                title_lines = wrap_text_to_width(title, inner_w, title_fsize, 0.65)

            def calc_stack_h(e_fz, emp_fz, t_fz, s_fz, t_lines):
                h_acc = 0
                if eyebrow: h_acc += e_fz + 18
                if emphasis: h_acc += emp_fz + 20
                if t_lines: h_acc += len(t_lines) * (t_fz * 1.18) + 12
                if sub: h_acc += s_fz * 1.3 + 10
                return h_acc

            stack_h = calc_stack_h(eyebrow_fsize, emphasis_fsize, title_fsize, sub_fsize, title_lines)
            if stack_h > inner_h * 0.95:
                scale_ratio = (inner_h * 0.90) / max(1, stack_h)
                eyebrow_fsize = max(14, int(eyebrow_fsize * scale_ratio))
                emphasis_fsize = max(44, int(emphasis_fsize * scale_ratio))
                title_fsize = max(24, int(title_fsize * scale_ratio))
                sub_fsize = max(16, int(sub_fsize * scale_ratio))
                title_lines = wrap_text_to_width(title, inner_w, title_fsize, 0.65) if title else []
                stack_h = calc_stack_h(eyebrow_fsize, emphasis_fsize, title_fsize, sub_fsize, title_lines)

            curr_y = box_y + (box_h - stack_h) // 2 + eyebrow_fsize + 10
            items = []

            if eyebrow:
                items.append(f'<text x="{center_x}" y="{curr_y}" font-family="{font_display}" font-size="{eyebrow_fsize}" font-weight="700" fill="#94A3B8" text-anchor="middle" letter-spacing="4">{esc(eyebrow).upper()}</text>')
                curr_y += eyebrow_fsize + 18

            if emphasis:
                items.append(f'<text x="{center_x}" y="{curr_y + emphasis_fsize * 0.85}" font-family="{font_display}" font-size="{emphasis_fsize}" font-weight="900" fill="{accent}" text-anchor="middle">{esc(emphasis)}</text>')
                curr_y += emphasis_fsize + 20

            if title_lines:
                for line in title_lines:
                    items.append(f'<text x="{center_x}" y="{curr_y + title_fsize * 0.85}" font-family="{font_display}" font-size="{title_fsize}" font-weight="900" fill="{fg}" text-anchor="middle">{esc(line).upper()}</text>')
                    curr_y += int(title_fsize * 1.18)
                curr_y += 10

            if sub and sub.lower() != title.lower():
                sub_lines = wrap_text_to_width(sub, inner_w, sub_fsize, 0.58)
                for line in sub_lines[:2]:
                    items.append(f'<text x="{center_x}" y="{curr_y + sub_fsize * 0.85}" font-family="{font_display}" font-size="{sub_fsize}" font-weight="600" fill="#CBD5E1" text-anchor="middle">{esc(line)}</text>')
                    curr_y += int(sub_fsize * 1.25)

            content_svg = "\n".join(items)

    elif component == "StatCard":
        label = str(props.get("label", "METRIC") or "METRIC").strip()
        value = str(props.get("value", "0") or "0").strip()
        caption = str(props.get("caption", "") or "").strip()
        inverted = props.get("inverted", False)

        card_bg = "#FFFFFF" if inverted else "#1E293B"
        card_fg = "#0F172A" if inverted else "#FFFFFF"
        val_color = "#0F172A" if inverted else accent

        val_fsize = 84 if is_landscape else 72
        lbl_fsize = 32 if is_landscape else 28
        cap_fsize = 22 if is_landscape else 20

        max_val_w = inner_w * 0.45
        if len(value) * val_fsize * 0.65 > max_val_w:
            val_fsize = max(40, int(max_val_w / (len(value) * 0.65)))

        lbl_max_w = inner_w * 0.48
        lbl_lines = wrap_text_to_width(label, lbl_max_w, lbl_fsize, 0.68)
        if len(lbl_lines) > 2:
            lbl_fsize = max(20, int(lbl_fsize * 0.8))
            lbl_lines = wrap_text_to_width(label, lbl_max_w, lbl_fsize, 0.68)

        stat_box_h = max(110, max(len(lbl_lines) * int(lbl_fsize * 1.2), val_fsize + 20) + 36)
        total_h = stat_box_h + (cap_fsize + 36 if caption else 0)

        start_y = box_y + (box_h - total_h) // 2
        items = []

        items.append(f'<rect x="{box_x + pad_x}" y="{start_y}" width="{inner_w}" height="{stat_box_h}" rx="16" fill="{card_bg}" stroke="#334155" stroke-width="2"/>')

        lbl_start_y = start_y + (stat_box_h - (len(lbl_lines) * int(lbl_fsize * 1.2))) // 2 + int(lbl_fsize * 0.85)
        for i, line in enumerate(lbl_lines):
            items.append(f'<text x="{box_x + pad_x + 32}" y="{lbl_start_y + i * int(lbl_fsize * 1.2)}" font-family="{font_display}" font-size="{lbl_fsize}" font-weight="800" fill="{card_fg}" letter-spacing="2">{esc(line).upper()}</text>')

        val_y = start_y + (stat_box_h // 2) + int(val_fsize * 0.35)
        items.append(f'<text x="{box_x + box_w - pad_x - 32}" y="{val_y}" font-family="{font_display}" font-size="{val_fsize}" font-weight="900" fill="{val_color}" text-anchor="end">{esc(value)}</text>')

        if caption:
            cap_y = start_y + stat_box_h + 18
            cap_lines = wrap_text_to_width(caption, inner_w - 40, cap_fsize, 0.60)
            cap_box_h = len(cap_lines) * int(cap_fsize * 1.25) + 16
            items.append(f'<rect x="{center_x - (inner_w * 0.4)}" y="{cap_y}" width="{inner_w * 0.8}" height="{cap_box_h}" rx="10" fill="#1E293B" stroke="#475569" stroke-width="1.5"/>')
            for ci, cline in enumerate(cap_lines):
                items.append(f'<text x="{center_x}" y="{cap_y + 14 + int(cap_fsize * 0.85) + ci * int(cap_fsize * 1.25)}" font-family="{font_display}" font-size="{cap_fsize}" font-weight="700" fill="#CBD5E1" text-anchor="middle" letter-spacing="1">{esc(cline).upper()}</text>')

        content_svg = "\n".join(items)

    elif component == "RuleCard":
        kicker = str(props.get("kicker", "BEST PRACTICE") or "BEST PRACTICE").strip()
        text = str(props.get("text", "") or "").strip()
        struck = str(props.get("struck", "") or "").strip()

        kicker_fsize = 20 if is_landscape else 18
        text_fsize = 46 if is_landscape else 38
        struck_fsize = 28 if is_landscape else 24

        text_lines = wrap_text_to_width(text, inner_w - 60, text_fsize, 0.65)
        if len(text_lines) > 3:
            text_fsize = 36 if is_landscape else 30
            text_lines = wrap_text_to_width(text, inner_w - 60, text_fsize, 0.65)

        struck_lines = wrap_text_to_width(struck, inner_w - 60, struck_fsize, 0.65) if struck else []

        card_pad = 28
        inner_card_w = inner_w
        inner_card_h = (kicker_fsize + 14) + (len(text_lines) * int(text_fsize * 1.18)) + (card_pad * 2)
        total_h = inner_card_h + ((len(struck_lines) * int(struck_fsize * 1.25) + 20) if struck else 0)

        if total_h > inner_h:
            s_ratio = (inner_h * 0.92) / total_h
            text_fsize = max(22, int(text_fsize * s_ratio))
            struck_fsize = max(18, int(struck_fsize * s_ratio))
            text_lines = wrap_text_to_width(text, inner_w - 60, text_fsize, 0.65)
            struck_lines = wrap_text_to_width(struck, inner_w - 60, struck_fsize, 0.65) if struck else []
            inner_card_h = (kicker_fsize + 14) + (len(text_lines) * int(text_fsize * 1.18)) + (card_pad * 2)
            total_h = inner_card_h + ((len(struck_lines) * int(struck_fsize * 1.25) + 20) if struck else 0)

        start_y = box_y + (box_h - total_h) // 2
        items = []

        items.append(f'<rect x="{box_x + pad_x}" y="{start_y}" width="{inner_card_w}" height="{inner_card_h}" rx="18" fill="#FFFFFF"/>')

        curr_y = start_y + card_pad + kicker_fsize
        items.append(f'<text x="{box_x + pad_x + 36}" y="{curr_y}" font-family="{font_mono}" font-size="{kicker_fsize}" font-weight="700" fill="#64748B" letter-spacing="4">{esc(kicker).upper()}</text>')
        curr_y += 18

        for line in text_lines:
            curr_y += int(text_fsize * 1.05)
            items.append(f'<text x="{box_x + pad_x + 36}" y="{curr_y}" font-family="{font_display}" font-size="{text_fsize}" font-weight="900" fill="#0F172A">{esc(line).upper()}</text>')

        if struck_lines:
            curr_y = start_y + inner_card_h + 20
            for sline in struck_lines:
                curr_y += int(struck_fsize * 1.15)
                s_width = len(sline) * struck_fsize * 0.65
                strike_start_x = center_x - (s_width / 2)
                items.append(f'<text x="{center_x}" y="{curr_y}" font-family="{font_display}" font-size="{struck_fsize}" font-weight="800" fill="#EF4444" text-anchor="middle" letter-spacing="2">{esc(sline).upper()}</text>')
                items.append(f'<line x1="{strike_start_x - 10}" y1="{curr_y - struck_fsize * 0.32}" x2="{strike_start_x + s_width + 10}" y2="{curr_y - struck_fsize * 0.32}" stroke="#EF4444" stroke-width="4" stroke-linecap="round"/>')

        content_svg = "\n".join(items)

    elif component == "StrikethroughReveal":
        lines = props.get("lines", [])
        if not lines: lines = [{"text": "DECISION"}]

        n_lines = len(lines)
        base_fsize = 36 if is_landscape else 30
        if n_lines > 3: base_fsize = 28 if is_landscape else 24

        header_fsize = 18
        items = [
            f'<text x="{center_x}" y="{box_y + pad_y + header_fsize}" font-family="{font_display}" font-size="{header_fsize}" font-weight="700" fill="#94A3B8" letter-spacing="3" text-anchor="middle">DECISION &amp; VERDICT</text>'
        ]

        avail_h = inner_h - (header_fsize + 30)
        row_h = avail_h // max(1, n_lines)
        curr_y = box_y + pad_y + header_fsize + 24

        for i, l in enumerate(lines):
            raw_text = str(l.get("text", "") or "").strip()
            struck = bool(l.get("struck", False))
            mark = str(l.get("mark", "none"))

            row_mid_y = curr_y + (row_h // 2)
            mark_svg = ""
            text_x = box_x + pad_x + 50

            if mark == "cross":
                mark_svg = f"""<g transform="translate({box_x + pad_x + 24}, {row_mid_y - 2})">
                  <circle cx="0" cy="0" r="15" fill="#EF4444" fill-opacity="0.25"/>
                  <path d="M-5,-5 L5,5 M5,-5 L-5,5" stroke="#EF4444" stroke-width="3" stroke-linecap="round"/>
                </g>"""
            elif mark == "check":
                mark_svg = f"""<g transform="translate({box_x + pad_x + 24}, {row_mid_y - 2})">
                  <circle cx="0" cy="0" r="15" fill="#10B981" fill-opacity="0.25"/>
                  <path d="M-6,0 L-2,4 L6,-4" stroke="#10B981" stroke-width="3" stroke-linecap="round" fill="none"/>
                </g>"""
            else:
                text_x = box_x + pad_x + 20

            max_text_w = inner_w - (70 if mark != "none" else 40)
            wrapped = wrap_text_to_width(raw_text, max_text_w, base_fsize, 0.65)
            text_color = "#64748B" if struck else "#FFFFFF"

            items.append(mark_svg)
            for li, wline in enumerate(wrapped):
                line_y = row_mid_y + (li * int(base_fsize * 1.15)) + int(base_fsize * 0.3)
                items.append(f'<text x="{text_x}" y="{line_y}" font-family="{font_display}" font-size="{base_fsize}" font-weight="800" fill="{text_color}">{esc(wline).upper()}</text>')
                if struck:
                    w_w = len(wline) * base_fsize * 0.65
                    items.append(f'<line x1="{text_x - 6}" y1="{line_y - base_fsize * 0.32}" x2="{text_x + w_w + 6}" y2="{line_y - base_fsize * 0.32}" stroke="#EF4444" stroke-width="3.5" stroke-linecap="round"/>')

            curr_y += row_h

        content_svg = "\n".join(items)

    elif component == "FlowDiagram":
        nodes = props.get("nodes", [])
        if not nodes: nodes = ["A", "B"]
        emphasize_last = props.get("emphasizeLast", True)
        n_count = len(nodes)

        items = [
            f'<text x="{center_x}" y="{box_y + pad_y + 16}" font-family="{font_display}" font-size="17" font-weight="700" fill="#94A3B8" letter-spacing="3" text-anchor="middle">ARCHITECTURE PIPELINE</text>'
        ]

        # Use horizontal only on genuinely wide slots (pip-bubble, graphic-only, blurred-behind) with <= 4 nodes
        use_horizontal = is_landscape and (n_count <= 4) and (inner_w >= 850) and ((inner_w / max(1, inner_h)) >= 1.5)

        if use_horizontal:
            arrow_w = 44
            total_arrow_w = (n_count - 1) * arrow_w
            avail_chip_w = inner_w - total_arrow_w - 48
            chip_w = min(280, avail_chip_w // n_count)
            total_w = n_count * chip_w + total_arrow_w
            chip_h = min(80, max(56, inner_h - 90))
            chip_y = box_y + (box_h - chip_h) // 2 + 10
            start_x = center_x - (total_w // 2)

            for i, node in enumerate(nodes):
                cx = start_x + i * (chip_w + arrow_w)
                is_emph = (i == n_count - 1 and emphasize_last)
                bg = accent if is_emph else "#1E293B"
                text_color = "#0F172A" if is_emph else "#FFFFFF"
                border_color = accent if is_emph else "#475569"

                node_str = esc(str(node)).upper()
                node_fsize = 20
                if len(node_str) * node_fsize * 0.65 > (chip_w - 24):
                    node_fsize = max(13, int((chip_w - 24) / (len(node_str) * 0.65)))

                items.append(f'<rect x="{cx}" y="{chip_y}" width="{chip_w}" height="{chip_h}" rx="14" fill="{bg}" stroke="{border_color}" stroke-width="2"/>')
                items.append(f'<text x="{cx + chip_w//2}" y="{chip_y + chip_h//2 + int(node_fsize * 0.35)}" font-family="{font_display}" font-size="{node_fsize}" font-weight="900" fill="{text_color}" text-anchor="middle">{node_str}</text>')
                if i < n_count - 1:
                    ax = cx + chip_w + 14
                    ay = chip_y + chip_h // 2
                    items.append(f'<polygon points="{ax},{ay - 7} {ax + 16},{ay} {ax},{ay + 7}" fill="#64748B"/>')

        else:
            # Vertical stack (split-left, split-right, portrait, or 5+ nodes)
            arrow_h = 32
            total_arrow_h = (n_count - 1) * arrow_h
            avail_h = inner_h - 50 - total_arrow_h
            chip_h = min(68, max(44, avail_h // n_count))
            chip_w = min(560, inner_w - 40)
            chip_x = center_x - (chip_w // 2)

            total_stack_h = (n_count * chip_h) + total_arrow_h
            start_y = box_y + pad_y + 34 + max(0, (inner_h - 34 - total_stack_h) // 2)

            for i, node in enumerate(nodes):
                cy = start_y + i * (chip_h + arrow_h)
                is_emph = (i == n_count - 1 and emphasize_last)
                bg = accent if is_emph else "#1E293B"
                text_color = "#0F172A" if is_emph else "#FFFFFF"
                border_color = accent if is_emph else "#475569"

                node_str = esc(str(node)).upper()
                node_fsize = 22
                if len(node_str) * node_fsize * 0.65 > (chip_w - 28):
                    node_fsize = max(14, int((chip_w - 28) / (len(node_str) * 0.65)))

                items.append(f'<rect x="{chip_x}" y="{cy}" width="{chip_w}" height="{chip_h}" rx="14" fill="{bg}" stroke="{border_color}" stroke-width="2"/>')
                items.append(f'<text x="{center_x}" y="{cy + chip_h//2 + int(node_fsize * 0.35)}" font-family="{font_display}" font-size="{node_fsize}" font-weight="900" fill="{text_color}" text-anchor="middle">{node_str}</text>')
                if i < n_count - 1:
                    ax = center_x
                    ay = cy + chip_h + 8
                    items.append(f'<polygon points="{ax - 7},{ay} {ax + 7},{ay} {ax},{ay + 16}" fill="#64748B"/>')

        content_svg = "\n".join(items)

    elif component == "TerminalMock":
        windows = props.get("windows", [])
        if not windows:
            windows = [{"title": "terminal-01", "lines": ["$ ossclip transcribe input.mp4", "> Processing Float16 CUDA..."]}]
        fan_out = str(props.get("fanOut", "") or "").strip()

        win = windows[0]
        w_title = str(win.get("title", "bash") or "bash").strip()
        w_lines = win.get("lines", [])

        term_fsize = 22 if is_landscape else 20
        win_w = inner_w
        win_h = inner_h - (40 if fan_out else 0)
        start_y = box_y + pad_y

        items = [
            f'<rect x="{box_x + pad_x}" y="{start_y}" width="{win_w}" height="{win_h}" rx="16" fill="#0B0F19" stroke="#334155" stroke-width="2"/>',
            f'<rect x="{box_x + pad_x}" y="{start_y}" width="{win_w}" height="42" rx="16" fill="#1E293B"/>',
            f'<rect x="{box_x + pad_x}" y="{start_y + 26}" width="{win_w}" height="16" fill="#1E293B"/>',
            f'<circle cx="{box_x + pad_x + 24}" cy="{start_y + 21}" r="6" fill="#EF4444"/>',
            f'<circle cx="{box_x + pad_x + 42}" cy="{start_y + 21}" r="6" fill="#F59E0B"/>',
            f'<circle cx="{box_x + pad_x + 60}" cy="{start_y + 21}" r="6" fill="#10B981"/>',
            f'<text x="{center_x}" y="{start_y + 27}" font-family="{font_mono}" font-size="15" fill="#94A3B8" text-anchor="middle">{esc(w_title)}</text>'
        ]

        curr_ly = start_y + 70
        for line in w_lines:
            l_str = str(line).strip()
            if l_str.startswith("$"):
                cmd_str = l_str[1:].strip()
                items.append(f'<text x="{box_x + pad_x + 28}" y="{curr_ly}" font-family="{font_mono}" font-size="{term_fsize}" font-weight="700" fill="{accent}">$ <tspan fill="#FFFFFF">{esc(cmd_str)}</tspan></text>')
            elif l_str.startswith(">"):
                items.append(f'<text x="{box_x + pad_x + 28}" y="{curr_ly}" font-family="{font_mono}" font-size="{term_fsize}" font-weight="600" fill="#38BDF8">&gt; <tspan fill="#CBD5E1">{esc(l_str[1:].strip())}</tspan></text>')
            else:
                items.append(f'<text x="{box_x + pad_x + 28}" y="{curr_ly}" font-family="{font_mono}" font-size="{term_fsize}" font-weight="500" fill="#94A3B8">{esc(l_str)}</text>')
            curr_ly += int(term_fsize * 1.5)

        if fan_out:
            fan_y = start_y + win_h + 30
            items.append(f'<polygon points="{center_x - 7},{fan_y - 14} {center_x + 7},{fan_y - 14} {center_x},{fan_y - 4}" fill="{accent}"/>')
            items.append(f'<text x="{center_x}" y="{fan_y + 14}" font-family="{font_mono}" font-size="18" font-weight="800" fill="{accent}" text-anchor="middle" letter-spacing="2">{esc(fan_out).upper()}</text>')

        content_svg = "\n".join(items)

    elif component == "ChatMock":
        keyword = str(props.get("keyword", "") or "").strip()
        messages = props.get("messages", [])

        if keyword:
            cta_text = f'"{keyword.upper()}"'
            cta_fsize = 64 if is_landscape else 54
            max_w = inner_w * 0.80
            if len(cta_text) * cta_fsize * 0.65 > max_w:
                cta_fsize = max(32, int(max_w / (len(cta_text) * 0.65)))

            b_w = len(cta_text) * cta_fsize * 0.65 + 60
            b_h = cta_fsize * 1.5 + 30
            bx = center_x - (b_w / 2)
            by = box_y + (box_h - b_h) // 2

            items = [
                f'<text x="{center_x}" y="{by - 24}" font-family="{font_display}" font-size="18" font-weight="700" fill="#94A3B8" letter-spacing="3" text-anchor="middle">COMMENT KEYWORD BELOW</text>',
                f'<rect x="{bx}" y="{by}" width="{b_w}" height="{b_h}" rx="24" fill="{accent}" stroke="#FFFFFF" stroke-width="2"/>',
                f'<text x="{center_x}" y="{by + b_h//2 + int(cta_fsize * 0.35)}" font-family="{font_display}" font-size="{cta_fsize}" font-weight="900" fill="#0F172A" text-anchor="middle">{esc(cta_text)}</text>'
            ]
            content_svg = "\n".join(items)

        else:
            if not messages: messages = [{"from": "user", "text": "Hello"}]
            items = [
                f'<text x="{center_x}" y="{box_y + pad_y + 18}" font-family="{font_display}" font-size="18" font-weight="700" fill="#94A3B8" letter-spacing="3" text-anchor="middle">DIRECT CONVERSATION</text>'
            ]
            chat_fsize = 26 if is_landscape else 22
            curr_by = box_y + pad_y + 50

            for i, msg in enumerate(messages[:3]):
                is_user = (msg.get("from") == "user")
                mtext = str(msg.get("text", "")).strip()

                wrapped = wrap_text_to_width(mtext, inner_w * 0.65, chat_fsize, 0.60)
                bub_w = min(inner_w * 0.75, max(len(l) for l in wrapped) * chat_fsize * 0.60 + 44)
                bub_h = len(wrapped) * int(chat_fsize * 1.25) + 24

                bub_x = (box_x + box_w - pad_x - bub_w) if is_user else (box_x + pad_x)
                bub_bg = accent if is_user else "#1E293B"
                bub_fg = "#0F172A" if is_user else "#FFFFFF"

                items.append(f'<rect x="{bub_x}" y="{curr_by}" width="{bub_w}" height="{bub_h}" rx="18" fill="{bub_bg}" stroke="#475569" stroke-width="1.5"/>')
                for li, lstr in enumerate(wrapped):
                    items.append(f'<text x="{bub_x + 22}" y="{curr_by + 16 + int(chat_fsize * 0.85) + li * int(chat_fsize * 1.25)}" font-family="{font_display}" font-size="{chat_fsize}" font-weight="700" fill="{bub_fg}">{esc(lstr)}</text>')

                curr_by += bub_h + 16

            content_svg = "\n".join(items)

    elif component in ["ScreenshotFrame", "BrowserFrame"]:
        label = str(props.get("label", "DASHBOARD") or "DASHBOARD").strip()
        items = [
            f'<rect x="{box_x + pad_x}" y="{box_y + pad_y}" width="{inner_w}" height="42" rx="14" fill="#1E293B"/>',
            f'<rect x="{box_x + pad_x}" y="{box_y + pad_y + 26}" width="{inner_w}" height="16" fill="#1E293B"/>',
            f'<circle cx="{box_x + pad_x + 24}" cy="{box_y + pad_y + 21}" r="6" fill="#EF4444"/>',
            f'<circle cx="{box_x + pad_x + 42}" cy="{box_y + pad_y + 21}" r="6" fill="#F59E0B"/>',
            f'<circle cx="{box_x + pad_x + 60}" cy="{box_y + pad_y + 21}" r="6" fill="#10B981"/>',
            f'<text x="{center_x}" y="{box_y + pad_y + 27}" font-family="{font_mono}" font-size="14" fill="#94A3B8" text-anchor="middle">app.preview.local</text>',
            f'<g transform="translate({box_x + pad_x + 36}, {box_y + pad_y + 65})">'
        ]
        skeleton_widths = [0.90, 0.75, 0.85, 0.60, 0.80]
        s_y = 20
        for i, w_frac in enumerate(skeleton_widths):
            line_w = int((inner_w - 72) * w_frac)
            color = "#334155" if i % 2 == 0 else "#1E293B"
            items.append(f'<rect x="0" y="{s_y}" width="{line_w}" height="20" rx="6" fill="{color}"/>')
            s_y += 36
        items.append('</g>')

        chip_w = len(label) * 16 + 40
        items.append(f"""<g transform="translate({box_x + box_w - pad_x - chip_w}, {box_y + box_h - pad_y - 44})">
          <rect x="0" y="0" width="{chip_w}" height="40" rx="10" fill="#FFFFFF"/>
          <text x="{chip_w // 2}" y="26" font-family="{font_display}" font-size="16" font-weight="900" fill="#0F172A" text-anchor="middle" letter-spacing="3">{esc(label).upper()}</text>
        </g>""")
        content_svg = "\n".join(items)

    elif component == "BulletList":
        title = str(props.get("title", "") or "").strip()
        items_data = props.get("items", [])
        if not items_data: items_data = ["FIRST POINT", "SECOND POINT"]

        b_fsize = 32 if is_landscape else 26
        n_items = len(items_data)
        if n_items > 3: b_fsize = 26 if is_landscape else 22

        items = []
        curr_y = box_y + pad_y + 24
        if title:
            items.append(f'<text x="{box_x + pad_x + 20}" y="{curr_y}" font-family="{font_display}" font-size="20" font-weight="700" fill="#94A3B8" letter-spacing="3">{esc(title).upper()}</text>')
            curr_y += 38

        for itm in items_data:
            wrapped = wrap_text_to_width(str(itm), inner_w - 80, b_fsize, 0.65)
            for li, wline in enumerate(wrapped):
                if li == 0:
                    bx = box_x + pad_x + 20
                    by = curr_y - int(b_fsize * 0.32)
                    items.append(f'<polygon points="{bx},{by - 7} {bx + 11},{by} {bx},{by + 7}" fill="{accent}"/>')
                items.append(f'<text x="{box_x + pad_x + 44}" y="{curr_y}" font-family="{font_display}" font-size="{b_fsize}" font-weight="800" fill="{fg}">{esc(wline).upper()}</text>')
                curr_y += int(b_fsize * 1.25)
            curr_y += 10

        content_svg = "\n".join(items)

    else:
        title = str(props.get("title", component) or component)
        content_svg = f'<text x="{center_x}" y="{box_y + box_h//2 + 10}" font-family="{font_display}" font-size="36" font-weight="900" fill="{fg}" text-anchor="middle">{esc(title).upper()}</text>'

    return f"""<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <filter id="shadow" x="-10%" y="-10%" width="120%" height="130%">
      <feDropShadow dx="0" dy="16" stdDeviation="24" flood-color="#000000" flood-opacity="0.85"/>
    </filter>
  </defs>
  <g {transform_attr} filter="url(#shadow)">
    <rect x="{box_x}" y="{box_y}" width="{box_w}" height="{box_h}" rx="22" fill="#0F172A" fill-opacity="0.95" stroke="#334155" stroke-width="2"/>
    {content_svg}
  </g>
</svg>"""

def render_cairo_png(svg_str, out_png, w=1920, h=1080):
    try:
        import cairosvg
        cairosvg.svg2png(bytestring=svg_str.encode('utf-8'), write_to=out_png, output_width=w, output_height=h)
        return True
    except Exception as e:
        print(f"Warning: Cairo rendering failed: {e}")
        return False

def build_ass_subtitles(caption_lines, cues, theme, out_ass_path, res_x=1920, res_y=1080, style_info=None):
    if not style_info: style_info = {}
    font_display = theme.get("fontDisplay", style_info.get("fontDisplay", "Montserrat"))
    font_display = font_display.replace("'", "").replace('"', '').split(',')[0].strip()
    accent_hex = theme.get("accent", style_info.get("accent", "#FFE600"))
    accent_ass = hex_to_ass_color(accent_hex)
    normal_ass = "&H00FFFFFF&"

    is_landscape = (res_x > res_y)
    font_size = style_info.get("fontSizeLandscape" if is_landscape else "fontSizePortrait", 46 if is_landscape else 54)
    outline_px = style_info.get("outlinePx", 4)
    shadow_px = style_info.get("shadowPx", 4)
    default_anchor = 0.80 if is_landscape else 0.70

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

    for line in caption_lines:
        words = line.get("words", [])
        if not words: continue
        for i, current_word in enumerate(words):
            start_t = current_word["start"]
            end_t = current_word["end"]
            start_str = format_ass_time(start_t)
            end_str = format_ass_time(end_t)

            active_anchor = default_anchor
            for c in cues:
                if c["start"] <= start_t <= c["end"]:
                    active_anchor = c.get("caption_anchor", default_anchor)
                    break

            sub_y = int(active_anchor * res_y)
            pos_tag = f"{{\\pos({res_x // 2},{sub_y})}}"

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
    parser = argparse.ArgumentParser(description="OSSClip High-Speed GPU Exporter with Full Remotion Layout Parity (Landscape & Portrait)")
    parser.add_argument("workdir")
    parser.add_argument("--out", "-o", default=None)
    parser.add_argument("--format", choices=["auto", "vertical", "original", "blur-backdrop"], default="auto")
    parser.add_argument("--style", choices=list(STYLE_PRESETS.keys()) + ["default"], default="hormozi")
    parser.add_argument("--graphics-style", default="auto")
    parser.add_argument("--bitrate", default="6M")
    parser.add_argument("--no-graphics", action="store_true")
    parser.add_argument("--no-captions", action="store_true")
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

    shm_dir = "/dev/shm" if os.path.exists("/dev/shm") else workdir

    settings = render_props.get("settings", {})
    prop_w = settings.get("width", 1920)
    prop_h = settings.get("height", 1080)

    if args.format == "auto":
        target_format = "vertical" if prop_h > prop_w else "original"
    else:
        target_format = args.format

    is_vertical = target_format in ["vertical", "blur-backdrop"] or (prop_h > prop_w)
    res_x = 1080 if is_vertical else prop_w
    res_y = 1920 if is_vertical else prop_h
    is_landscape = not is_vertical

    style_info = STYLE_PRESETS.get(args.style, {})
    theme = {**render_props.get("theme", {}), **style_info, **overrides.get("theme", {})}

    if overrides.get("captionsHidden") is True:
        args.no_captions = True

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

    spans = render_props.get("spans", [])
    user_cuts = overrides.get("cuts", [])
    if user_cuts:
        spans = subtract_user_cuts(spans, user_cuts)

    output_duration = sum(max(0, s["srcOut"] - s["srcIn"]) for s in spans)
    if output_duration <= 0: output_duration = 120.0

    source_video = None
    if os.path.exists(os.path.join(workdir, "production.json")):
        source_video = json.load(open(os.path.join(workdir, "production.json"))).get("source", {}).get("path")
    if not source_video or not os.path.exists(source_video):
        for c in ["mezzanine.mp4", "mezzanine-content.mp4"]:
            p = os.path.join(workdir, c)
            if os.path.exists(p) and os.path.getsize(p) > 1000:
                source_video = p; break

    graphic_overlays = []
    blur_intervals = []
    split_left_intervals = []
    split_right_intervals = []
    video_top_intervals = []
    graphic_only_intervals = []
    pip_bubble_intervals = []

    if not args.no_graphics and args.graphics_style != "none":
        scene_cues = render_props.get("sceneCues", [])
        scene_overrides = overrides.get("scenes", {})
        pid = os.getpid()

        for idx, cue in enumerate(scene_cues):
            cue_id = cue.get("id", f"scene-{idx}")
            sc_override = scene_overrides.get(cue_id, {}) or scene_overrides.get(f"scene-{idx}", {})

            if sc_override.get("hidden") is True:
                continue

            comp = sc_override.get("component", cue.get("component"))
            if not comp or comp == "None" or (cue.get("kind") == "plain" and not sc_override.get("component")):
                continue

            comp = normalize_component(comp)
            props = {**cue.get("props", {}), **sc_override.get("props", {})}
            elem_transforms = sc_override.get("elements", {})
            for elem_id, elem_data in elem_transforms.items():
                if isinstance(elem_data, dict) and "text" in elem_data and elem_data["text"] is not None:
                    props[elem_id] = elem_data["text"]

            card_transform = elem_transforms.get("card", {}) or elem_transforms.get("root", {})
            dx = card_transform.get("dx", 0)
            dy = card_transform.get("dy", 0)
            scale = card_transform.get("scale", 1.0)

            start_t = cue.get("startSec", 0)
            end_t = cue.get("endSec", start_t + 5)
            if sc_override.get("timing"):
                timing = sc_override["timing"]
                start_t = timing.get("srcStart", timing.get("startSec", start_t))
                end_t = timing.get("srcEnd", timing.get("endSec", end_t))

            layout = normalize_layout(sc_override.get("layout", cue.get("layout", "lower-third")))
            geom = get_stage_geometry(layout, is_landscape=is_landscape, w=res_x, h=res_y)

            cond = f"between(t,{start_t:.3f},{end_t:.3f})"
            if geom["video_mode"] == "blur":
                blur_intervals.append(cond)
            elif geom["video_mode"] == "split-left":
                split_left_intervals.append(cond)
            elif geom["video_mode"] == "split-right":
                split_right_intervals.append(cond)
            elif geom["video_mode"] == "video-top":
                video_top_intervals.append(cond)
            elif geom["video_mode"] == "graphic-only":
                graphic_only_intervals.append(cond)
            elif geom["video_mode"] == "pip-bubble":
                pip_bubble_intervals.append(cond)

            svg = generate_scene_svg(
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
            if render_cairo_png(svg, png_path, w=res_x, h=res_y):
                graphic_overlays.append({
                    "path": png_path,
                    "start": start_t,
                    "end": end_t,
                    "comp": comp,
                    "layout": layout,
                    "caption_anchor": geom["caption_anchor"]
                })

    ass_path = os.path.join(shm_dir, f"subtitles_custom_{os.getpid()}.ass")
    if not args.no_captions:
        build_ass_subtitles(caption_lines, graphic_overlays, theme, ass_path, res_x=res_x, res_y=res_y, style_info=style_info)

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

    filter_chains = []

    # 1. Base scaling
    base_vf = []
    if select_filter != "1":
        base_vf.append(f"select='{select_filter}',setpts=N/FRAME_RATE/TB")
    if is_vertical:
        base_vf.append(f"crop=ih*9/16:ih:(iw-ih*9/16)/2:0,scale=1080:1920")
    else:
        base_vf.append(f"scale={res_x}:{res_y}")

    # 2. Dynamic Blur FX (Blurred-Behind)
    if blur_intervals:
        blur_enable = "+".join(blur_intervals)
        base_vf.append(f"boxblur=15:1:enable='{blur_enable}'")
        base_vf.append(f"eq=brightness=-0.12:enable='{blur_enable}'")

    filter_chains.append(f"[0:v]{','.join(base_vf)}[v_scaled]")
    curr_stage = "v_scaled"

    # 3. Dynamic Video-Top FX
    if video_top_intervals:
        vt_cond = "+".join(video_top_intervals)
        top_h = 806 if is_vertical else (res_y // 2)
        filter_chains.append(f"[{curr_stage}]split=2[v_norm_vt][v_crop_vt]")
        filter_chains.append(f"color=c=#0B0F19:s={res_x}x{res_y}:d={output_duration:.2f}[bg_vt]")
        filter_chains.append(f"[v_crop_vt]crop={res_x}:{top_h}:0:0[v_top_half]")
        filter_chains.append(f"[bg_vt][v_top_half]overlay=x=0:y=0[stage_vt]")
        filter_chains.append(f"[v_norm_vt][stage_vt]overlay=enable='{vt_cond}'[v_stage_vt]")
        curr_stage = "v_stage_vt"

    # 4. Dynamic Graphic-Only FX (Video hidden, solid dark stage canvas)
    if graphic_only_intervals:
        go_cond = "+".join(graphic_only_intervals)
        filter_chains.append(f"color=c=#0B0F19:s={res_x}x{res_y}:d={output_duration:.2f}[bg_go]")
        filter_chains.append(f"[{curr_stage}][bg_go]overlay=enable='{go_cond}'[v_stage_go]")
        curr_stage = "v_stage_go"

    # 5. Dynamic Pip-Bubble FX (Circular video bubble against dark stage)
    if pip_bubble_intervals:
        pb_cond = "+".join(pip_bubble_intervals)
        pip_diam = 340 if is_vertical else 320
        pip_x = (res_x - pip_diam) // 2 if is_vertical else (res_x - pip_diam - 80)
        pip_y = 1260 if is_vertical else (res_y - pip_diam - 80)
        filter_chains.append(f"[{curr_stage}]split=2[v_norm_pb][v_crop_pb]")
        filter_chains.append(f"color=c=#0B0F19:s={res_x}x{res_y}:d={output_duration:.2f}[bg_pb]")
        filter_chains.append(f"[v_crop_pb]crop='min(iw,ih)':'min(iw,ih)':'(iw-min(iw,ih))/2':'(ih-min(iw,ih))/2',scale={pip_diam}:{pip_diam}[v_pip_box]")
        filter_chains.append(f"[bg_pb][v_pip_box]overlay=x={pip_x}:y={pip_y}[stage_pb]")
        filter_chains.append(f"[v_norm_pb][stage_pb]overlay=enable='{pb_cond}'[v_stage_pb]")
        curr_stage = "v_stage_pb"

    # 6. Dynamic Split-Left FX
    if split_left_intervals:
        sl_cond = "+".join(split_left_intervals)
        if is_landscape:
            half_w = res_x // 2
            filter_chains.append(f"[{curr_stage}]split=2[v_norm_sl][v_crop_sl]")
            filter_chains.append(f"color=c=#0B0F19:s={res_x}x{res_y}:d={output_duration:.2f}[bg_sl]")
            filter_chains.append(f"[v_crop_sl]crop={half_w}:{res_y}:{half_w // 2}:0[video_half_l]")
            filter_chains.append(f"[bg_sl][video_half_l]overlay=x=0:y=0[split_canvas_l]")
            filter_chains.append(f"[v_norm_sl][split_canvas_l]overlay=enable='{sl_cond}'[v_stage_sl]")
            curr_stage = "v_stage_sl"
        else:
            half_h = res_y // 2
            filter_chains.append(f"[{curr_stage}]split=2[v_norm_sl][v_crop_sl]")
            filter_chains.append(f"color=c=#0B0F19:s={res_x}x{res_y}:d={output_duration:.2f}[bg_sl]")
            filter_chains.append(f"[v_crop_sl]crop={res_x}:{half_h}:0:0[video_half_l]")
            filter_chains.append(f"[bg_sl][video_half_l]overlay=x=0:y=0[split_canvas_l]")
            filter_chains.append(f"[v_norm_sl][split_canvas_l]overlay=enable='{sl_cond}'[v_stage_sl]")
            curr_stage = "v_stage_sl"

    # 7. Dynamic Split-Right FX
    if split_right_intervals:
        sr_cond = "+".join(split_right_intervals)
        if is_landscape:
            half_w = res_x // 2
            filter_chains.append(f"[{curr_stage}]split=2[v_norm_sr][v_crop_sr]")
            filter_chains.append(f"color=c=#0B0F19:s={res_x}x{res_y}:d={output_duration:.2f}[bg_sr]")
            filter_chains.append(f"[v_crop_sr]crop={half_w}:{res_y}:{half_w // 2}:0[video_half_r]")
            filter_chains.append(f"[bg_sr][video_half_r]overlay=x={half_w}:y=0[split_canvas_r]")
            filter_chains.append(f"[v_norm_sr][split_canvas_r]overlay=enable='{sr_cond}'[v_stage_sr]")
            curr_stage = "v_stage_sr"
        else:
            half_h = res_y // 2
            filter_chains.append(f"[{curr_stage}]split=2[v_norm_sr][v_crop_sr]")
            filter_chains.append(f"color=c=#0B0F19:s={res_x}x{res_y}:d={output_duration:.2f}[bg_sr]")
            filter_chains.append(f"[v_crop_sr]crop={res_x}:{half_h}:0:0[video_half_r]")
            filter_chains.append(f"[bg_sr][video_half_r]overlay=x=0:y=0[split_canvas_r]")
            filter_chains.append(f"[v_norm_sr][split_canvas_r]overlay=enable='{sr_cond}'[v_stage_sr]")
            curr_stage = "v_stage_sr"

    # 8. Overlays (Graphic Cards from Cairo)
    last_v = curr_stage
    for i, g in enumerate(graphic_overlays):
        next_v = f"v_ov_{i}"
        filter_chains.append(f"[{last_v}][{i+1}:v]overlay=enable='between(t,{g['start']},{g['end']})':format=auto[{next_v}]")
        last_v = next_v

    # 9. Burn-in Subtitles with Dynamic Anchors
    ass_filter = f",ass={ass_path}" if not args.no_captions else ""
    filter_chains.append(f"[{last_v}]null{ass_filter}[outv]")

    cmd_filter = ["-filter_complex", "; ".join(filter_chains), "-map", "[outv]", "-map", "0:a"]

    if select_filter != "1":
        cmd_filter += ["-af", f"aselect='{select_filter}',asetpts=N/SR/TB"]

    gpu_nvenc_flags = [
        "-shortest",
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

    print(f"🎬 Starting High-Speed Cairo GPU Export (Tesla T4 NVENC with Complete Remotion Layout Parity)...")
    if graphic_overlays:
        print(f"✨ Active Stages ({'Portrait 9:16' if is_vertical else 'Landscape 16:9'}):")
        if video_top_intervals:
            print(f"   • Video-Top Stage Active ({len(video_top_intervals)} cues): Video on Top, Dark Stage at Bottom")
        if graphic_only_intervals:
            print(f"   • Graphic-Only Stage Active ({len(graphic_only_intervals)} cues): Video hidden, Full Dark Stage")
        if pip_bubble_intervals:
            print(f"   • PiP-Bubble Stage Active ({len(pip_bubble_intervals)} cues): Floating Circular Video Bubble + Card")
        if split_left_intervals:
            print(f"   • Split-Left Stage Active ({len(split_left_intervals)} cues): Video Left/Top, Dark Stage Right/Bottom")
        if split_right_intervals:
            print(f"   • Split-Right Stage Active ({len(split_right_intervals)} cues): Video Right/Top, Dark Stage Left/Bottom")
        if blur_intervals:
            print(f"   • Blurred-Behind Stage Active ({len(blur_intervals)} cues): Dynamic Video Blur + Centered Card")
        for idx, g in enumerate(graphic_overlays):
            print(f"   [{idx+1}] {g['comp']} ({g['layout']}) at {g['start']:.2f}s -> {g['end']:.2f}s (anchor: {g['caption_anchor']})")
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

# Patch edit.ts to wire UI Render button to GPU exporter with Remotion layout parity
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
# Patch produce-wizard.ts to update prompt copy and make High-Speed GPU Engine the primary choice
replace_in_file(
    "/tools/node/lib/node_modules/ossclip/src/interactive/produce-wizard.ts",
    '''        message: "Render now, or review the cut first?",
        initialValue: "review",
        options: [
          {
            value: "review",
            label: "Review the cut first",
            hint: "opens the editor; render from its button",
          },
          { value: "render", label: "Render now", hint: "straight to a finished file" },
        ],''',
    '''        message: "Render now, or review the cut first?",
        initialValue: "render",
        options: [
          {
            value: "render",
            label: "Render now (High-Speed GPU Engine)",
            hint: "ultra-fast export via Tesla T4 NVENC (~30s, no Remotion)",
          },
          {
            value: "review",
            label: "Review the cut first in Web Editor",
            hint: "opens editor to adjust text, layouts, graphics",
          },
        ],'''
)

# Patch produce.ts to run ossclip-gpu-render directly during the render phase
replace_in_file(
    "/tools/node/lib/node_modules/ossclip/src/produce.ts",
    '''    await phases.time("render", () =>
      renderProduction(props, {
        publicDir: dirname(renderVideo),
        outPath: rawPath,''',
    '''    await phases.time("render", async () => {
      if (existsSync("/usr/local/bin/ossclip-gpu-render")) {
        if (renderHud) renderHud.stop();
        const { spawnSync } = await import("node:child_process");
        const workdir = dirname(renderVideo);
        console.log(`\n⚡ Rendering with High-Speed GPU Engine (Tesla T4 NVENC)...`);
        const res = spawnSync(
          "/usr/local/bin/ossclip-gpu-render",
          [workdir, "--format", "auto", "--out", rawPath],
          { stdio: "inherit" }
        );
        if (res.status !== 0) {
          throw new Error(`GPU render failed with exit code ${res.status}`);
        }
        return;
      }
      return renderProduction(props, {
        publicDir: dirname(renderVideo),
        outPath: rawPath,'''
)
replace_in_file(
    "/tools/node/lib/node_modules/ossclip/src/produce.ts",
    '''        },
      }),
    );''',
    '''        },
      });
    });'''
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

# Patch produce-wizard.ts so AI graphics prompt is explicit, defaults to No, and offers Cairo styles
replace_in_file(
    "/tools/node/lib/node_modules/ossclip/src/interactive/produce-wizard.ts",
    '''  const graphics = unwrap(
    await confirm({ message: "Plan title cards and graphics with an LLM?", initialValue: false }),
  ) as boolean;''',
    '''  const graphics = unwrap(
    await confirm({ message: "Create graphics with AI? (No = clean cut video without graphic overlays)", initialValue: false }),
  ) as boolean;

  const graphicsStyle = graphics
    ? (unwrap(
        await select({
          message: "Select Cairo AI graphics style:",
          options: [
            { value: "tokyo-night", label: "Tokyo Night Terminal", hint: "macOS Code & CLI Window" },
            { value: "linear", label: "Linear Obsidian", hint: "Cloud & Data Architecture Flow" },
            { value: "stripe", label: "Stripe Clean Paper", hint: "High-Contrast Comparison Matrix" },
            { value: "vercel", label: "Vercel Geist Dark", hint: "Performance Metrics & KPI Cards" },
            { value: "auto", label: "Auto Match", hint: "AI chooses based on speech topic" },
          ],
        }),
      ) as string)
    : undefined;'''
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

# Patch FlowDiagram.tsx and fit.ts for centered text, symmetric spacing, and slot auto-scaling
replace_in_file(
    "/tools/node/lib/node_modules/ossclip/node_modules/@ossclip/scenes/src/fit.ts",
    '''export const MIN_ROW_FONT = 26;
export const MIN_STACK_FONT = 22;''',
    '''export const MIN_ROW_FONT = 16;
export const MIN_STACK_FONT = 18;'''
)
replace_in_file(
    "/tools/node/lib/node_modules/ossclip/node_modules/@ossclip/scenes/src/fit.ts",
    '''const MAX_STACK_FONT = 76;''',
    '''const MAX_STACK_FONT = 52;'''
)
replace_in_file(
    "/tools/node/lib/node_modules/ossclip/node_modules/@ossclip/scenes/src/fit.ts",
    '''  const CHAR_W = 0.78;
  const CHIP_PAD = 1.6;
  const ARROW_W = 1.8;
  const budget = widthPx - 20; // root padding "0 10px"

  const rowFont = Math.floor(
    Math.min(budget / (CHAR_W * chars + CHIP_PAD * n + ARROW_W * (n - 1)), heightPx / CHIP_H),
  );
  const longest = Math.max(...nodes.map((node) => node.length), 1);
  const stackUnits = CHIP_H * n + ARROW_ROW_H * (n - 1);
  const stackFont = Math.floor(
    Math.min(budget / (CHAR_W * longest + CHIP_PAD), heightPx / stackUnits, MAX_STACK_FONT),
  );

  const rowFits = rowFont >= MIN_ROW_FONT;
  const stackFits = stackFont >= MIN_STACK_FONT;
  if (!stackFits) return { mode: "row", fontSize: Math.max(MIN_ROW_FONT, rowFont) };
  if (!rowFits) return { mode: "stack", fontSize: stackFont };
  // Both are legible — take the one that uses the slot.
  return rowFont * CHIP_H >= stackFont * stackUnits
    ? { mode: "row", fontSize: rowFont }
    : { mode: "stack", fontSize: stackFont };''',
    '''  const CHAR_W = 0.72;
  const CHIP_PAD = 1.6;
  const ARROW_W = 1.5;
  const budget = Math.max(100, widthPx - 56);
  const availHeight = Number.isFinite(heightPx) ? heightPx : 400;

  const rowFont = Math.floor(
    Math.min(budget / (CHAR_W * chars + CHIP_PAD * n + ARROW_W * (n - 1)), availHeight / CHIP_H),
  );
  const longest = Math.max(...nodes.map((node) => node.length), 1);
  const stackUnits = CHIP_H * n + ARROW_ROW_H * (n - 1);
  const stackFont = Math.floor(
    Math.min(budget / (CHAR_W * longest + CHIP_PAD), availHeight / stackUnits, MAX_STACK_FONT),
  );

  const preferStack = (widthPx < 740 && n >= 3) || (widthPx < availHeight * 1.15 && n >= 3);
  if (preferStack && stackFont >= MIN_STACK_FONT) {
    return { mode: "stack", fontSize: Math.min(36, stackFont) };
  }

  const rowFits = rowFont >= MIN_ROW_FONT;
  const stackFits = stackFont >= MIN_STACK_FONT;
  if (!stackFits) return { mode: "row", fontSize: Math.max(14, Math.min(48, rowFont)) };
  if (!rowFits) return { mode: "stack", fontSize: Math.max(16, Math.min(36, stackFont)) };

  return rowFont * CHIP_H >= stackFont * stackUnits
    ? { mode: "row", fontSize: Math.min(44, rowFont) }
    : { mode: "stack", fontSize: Math.min(36, stackFont) };'''
)
replace_in_file(
    "/tools/node/lib/node_modules/ossclip/node_modules/@ossclip/scenes/src/components/FlowDiagram.tsx",
    '''        borderRadius: theme.radiusPx / 2,
        padding: `${fontSize * 0.55}px ${fontSize * 0.8}px`,''',
    '''        borderRadius: Math.max(8, theme.radiusPx / 2),
        padding: `${Math.max(6, fontSize * 0.45)}px ${Math.max(14, fontSize * 0.8)}px`,'''
)
replace_in_file(
    "/tools/node/lib/node_modules/ossclip/node_modules/@ossclip/scenes/src/components/FlowDiagram.tsx",
    '''        whiteSpace: "nowrap",
        fontFamily: theme.fontDisplay,''',
    '''        whiteSpace: "nowrap",
        textAlign: "center",
        display: "inline-flex",
        alignItems: "center",
        justifyContent: "center",
        fontFamily: theme.fontDisplay,
        boxSizing: "border-box",
        maxWidth: "100%",
        flexShrink: 0,'''
)
replace_in_file(
    "/tools/node/lib/node_modules/ossclip/node_modules/@ossclip/scenes/src/components/FlowDiagram.tsx",
    '''        lineHeight: 1,
        paddingLeft: down ? 0 : fontSize * 0.55,''',
    '''        lineHeight: 1,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        flexShrink: 0,'''
)
replace_in_file(
    "/tools/node/lib/node_modules/ossclip/node_modules/@ossclip/scenes/src/components/FlowDiagram.tsx",
    '''    <div
      style={{
        display: "flex",
        flexDirection: row ? "row" : "column",
        flexWrap: "nowrap",
        alignItems: "center",
        justifyContent: "center",
        gap: row ? 0 : fontSize * 0.45,
        padding: "0 10px",
      }}
    >
      {props.nodes.map((node, i) => (
        // Arrow and its target chip are ONE flex item, and the arrow enters
        // AFTER its chip — an arrow into nothing is impossible in either shape.
        <div
          key={i}
          style={{
            display: "flex",
            flexDirection: row ? "row" : "column",
            alignItems: "center",
            gap: row ? fontSize * 0.55 : fontSize * 0.45,
          }}
        >
          {i > 0 ? <Arrow delay={i * 6 + 3} fontSize={fontSize} theme={theme} down={!row} /> : null}
          <Chip
            text={node}
            emphasized={props.emphasizeLast && i === props.nodes.length - 1}
            delay={i * 6}
            fontSize={fontSize}
            theme={theme}
            editId={`node-${i}`}
            edits={edits}
          />
        </div>
      ))}
    </div>''',
    '''    <div
      style={{
        display: "flex",
        flexDirection: row ? "row" : "column",
        flexWrap: "nowrap",
        alignItems: "center",
        justifyContent: "center",
        gap: row ? Math.max(12, fontSize * 0.55) : Math.max(10, fontSize * 0.45),
        padding: row ? "0 28px" : "14px 20px",
        width: "100%",
        height: "100%",
        boxSizing: "border-box",
      }}
    >
      {props.nodes.map((node, i) => (
        <React.Fragment key={i}>
          {i > 0 ? <Arrow delay={i * 6 + 3} fontSize={fontSize} theme={theme} down={!row} /> : null}
          <Chip
            text={node}
            emphasized={props.emphasizeLast && i === props.nodes.length - 1}
            delay={i * 6}
            fontSize={fontSize}
            theme={theme}
            editId={`node-${i}`}
            edits={edits}
          />
        </React.Fragment>
      ))}
    </div>'''
)
PYEOF


# Patch scene-registry.ts to expand altLayouts for all components
replace_in_file(
    "/tools/node/lib/node_modules/ossclip/node_modules/@ossclip/core/src/scene-registry.ts",
    '    defaultLayout: "pip-bubble",\n    altLayouts: [],',
    '    defaultLayout: "pip-bubble",\n    altLayouts: ["blurred-behind", "lower-third", "split-left", "split-right", "video-top", "graphic-only"],'
)
replace_in_file(
    "/tools/node/lib/node_modules/ossclip/node_modules/@ossclip/core/src/scene-registry.ts",
    '    defaultLayout: "video-top",\n    altLayouts: ["blurred-behind"],\n    whenToUse: "One striking metric',
    '    defaultLayout: "video-top",\n    altLayouts: ["blurred-behind", "lower-third", "split-left", "split-right", "graphic-only"],\n    whenToUse: "One striking metric'
)
replace_in_file(
    "/tools/node/lib/node_modules/ossclip/node_modules/@ossclip/core/src/scene-registry.ts",
    '    defaultLayout: "video-top",\n    altLayouts: ["blurred-behind"],\n    whenToUse:\n      "A prescriptive takeaway',
    '    defaultLayout: "video-top",\n    altLayouts: ["blurred-behind", "lower-third", "split-left", "split-right", "graphic-only"],\n    whenToUse:\n      "A prescriptive takeaway'
)
replace_in_file(
    "/tools/node/lib/node_modules/ossclip/node_modules/@ossclip/core/src/scene-registry.ts",
    '    defaultLayout: "blurred-behind",\n    altLayouts: ["graphic-only"],\n    whenToUse:\n      "Negation/contrast beat',
    '    defaultLayout: "blurred-behind",\n    altLayouts: ["graphic-only", "split-left", "split-right", "lower-third"],\n    whenToUse:\n      "Negation/contrast beat'
)
replace_in_file(
    "/tools/node/lib/node_modules/ossclip/node_modules/@ossclip/core/src/scene-registry.ts",
    '    defaultLayout: "graphic-only",\n    altLayouts: [],\n    whenToUse: "A causal chain',
    '    defaultLayout: "graphic-only",\n    altLayouts: ["blurred-behind", "split-left", "split-right", "video-top"],\n    whenToUse: "A causal chain'
)
replace_in_file(
    "/tools/node/lib/node_modules/ossclip/node_modules/@ossclip/core/src/scene-registry.ts",
    '    defaultLayout: "graphic-only",\n    altLayouts: [],\n    whenToUse: "Anything about running',
    '    defaultLayout: "graphic-only",\n    altLayouts: ["blurred-behind", "split-left", "split-right"],\n    whenToUse: "Anything about running'
)
replace_in_file(
    "/tools/node/lib/node_modules/ossclip/node_modules/@ossclip/core/src/scene-registry.ts",
    '    defaultLayout: "video-top",\n    altLayouts: ["blurred-behind"],\n    whenToUse: "Reference to a document',
    '    defaultLayout: "video-top",\n    altLayouts: ["blurred-behind", "split-left", "split-right", "lower-third"],\n    whenToUse: "Reference to a document'
)
replace_in_file(
    "/tools/node/lib/node_modules/ossclip/node_modules/@ossclip/core/src/scene-registry.ts",
    '    defaultLayout: "blurred-behind",\n    altLayouts: ["graphic-only"],\n    whenToUse:\n      "An ENUMERATION',
    '    defaultLayout: "blurred-behind",\n    altLayouts: ["graphic-only", "split-left", "split-right", "video-top"],\n    whenToUse:\n      "An ENUMERATION'
)

# Patch framing.ts so all 8 layouts are supported in landscape
replace_in_file(
    "/tools/node/lib/node_modules/ossclip/node_modules/@ossclip/core/src/framing.ts",
    '''export const LANDSCAPE_LAYOUTS: readonly Layout[] = [
  "full-bleed",
  "blurred-behind",
  "lower-third",
  "split-left",
  "split-right",
];

export function landscapeLayout(layout: Layout): Layout {
  if (LANDSCAPE_LAYOUTS.includes(layout)) return layout;
  return layout === "graphic-only" ? "blurred-behind" : "split-left";
}''',
    '''export const LANDSCAPE_LAYOUTS: readonly Layout[] = [
  "full-bleed",
  "blurred-behind",
  "lower-third",
  "split-left",
  "split-right",
  "graphic-only",
  "pip-bubble",
  "video-top",
];

export function landscapeLayout(layout: Layout): Layout {
  if (LANDSCAPE_LAYOUTS.includes(layout)) return layout;
  return "split-left";
}'''
)

# Patch beats.ts for comprehensive layout selection guidance in AI producer prompt
replace_in_file(
    "/tools/node/lib/node_modules/ossclip/node_modules/@ossclip/core/src/producer/beats.ts",
    '- VARIETY: never the same component twice in a row, and prefer a component you have NOT used yet in this video — reuse a treatment only when the beat genuinely calls for it. A repeat reads as a template.',
    '''- VARIETY: never the same component or layout twice in a row, and prefer components and layouts you have NOT used yet in this video — reuse a treatment only when the beat genuinely calls for it. A repeat reads as a template.
- LAYOUT SELECTION RULES:
  • "pip-bubble": Speaker's face in a floating circular bubble, graphic/headline prominently above. Best for TitleCard / big hook statement.
  • "graphic-only": Full visual focus on diagrams, code, architecture, or workflows with video hidden behind stage backdrop. Best for FlowDiagram, TerminalMock.
  • "blurred-behind": Dynamic background blur with dimmed video behind high-contrast cards. Best for StrikethroughReveal, BulletList, ChatMock, StatCard.
  • "split-left" / "split-right": Speaker on one side/top and graphic card on the other side/bottom. Great for side-by-side explanations, comparisons, and walkthroughs.
  • "video-top": Speaker in top portion of frame with punchy card in bottom stage canvas. Best for StatCard, RuleCard, ScreenshotFrame.
  • "lower-third": Unobtrusive broadcast graphic in the lower area while speaker remains full screen.
  • "full-bleed": Full-frame video with floating graphic card overlay.'''
)

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


