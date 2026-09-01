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

### 6. Maximum RAM & GPU Concurrency Architecture ⚡
To maximize hardware utilization on Linux & Google Colab (Tesla T4 with 15GB VRAM and 12GB RAM):
* **GPU Whisper ASR:**
  * Enabled Silero VAD (`vad_filter=True`, `min_silence_duration_ms=400`) to drop silent frames before speech decoding.
  * Increased batch size to `batch_size=24` across GPU Tensor Cores with Float16 compute.
  * Transcription speed: **1.47 seconds for a 94-second video (64x realtime)**.
* **RAM-Disk Buffering (`/dev/shm`):**
  * All intermediate vector graphics (Cairo PNGs) and ASS subtitle files are generated in `/dev/shm` (shared RAM) at **10+ GB/s** memory bandwidth, completely bypassing disk I/O.
* **Tesla T4 NVENC Hardware Pipelining:**
  * `-surfaces 32`: Pre-allocates 32 hardware surfaces in GPU VRAM so NVENC never stalls waiting for frames.
  * `-rc-lookahead 16`: Lookahead rate control buffered directly in GPU memory.
  * `-threads 4 -filter_threads 2`: Parallel demuxing and filter-complex execution.
  * `-preset p4`: Optimal throughput (~42+ FPS on multi-overlay compositions).

### Options

| Flag | Values | Description |
|---|---|---|
| `--format` | `vertical`, `original`, `blur-backdrop` | Aspect ratio framing (`vertical` = 9:16 vertical crop for Shorts/Reels/TikTok; `original` = 16:9 widescreen). |
| `--style` | `hormozi`, `mrbeast`, `cyber`, `pill`, `clean` | Viral caption style preset. |
| `--no-graphics` | (flag) | Disables all AI scene cards, rule cards, diagrams, and graphic overlays. |
| `--graphics-style` | `tokyo-night`, `linear`, `stripe`, `vercel`, `auto`, `none` | High-contrast Cairo vector graphic overlay (default: `auto`). |
| `--no-captions` | (flag) | Disables burned-in subtitles for clean video export. |
| `--out`, `-o` | `<path.mp4>` | Output video file path. |
| `--bitrate` | `6M`, `8M` | Video bitrate (default: `6M`). |

### Cairo AI Graphics Presets

- **`tokyo-night`**: macOS Terminal & Code Window with JetBrains Mono syntax highlighting and traffic lights.
- **`linear`**: Modern Silicon Valley obsidian architecture and cloud pipeline flow with glowing active node.
- **`stripe`**: High-contrast pure white card with comparison matrix badges (rejected vs best practice).
- **`vercel`**: Minimalist pitch-black card with 52pt bold metrics, benchmark counters, and green delta tags.
- **`none`**: Clean cut video with zero graphic overlays.

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

### Supported Stage Layouts & AI Selection

The AI Producer automatically selects and alternates layouts to maintain viral pacing and visual hierarchy:

| Layout | Visual Composition | Best Suited Component |
|---|---|---|
| **`pip-bubble`** | Floating circular face bubble with headline above. | `TitleCard` (Hook & Core Statements) |
| **`graphic-only`** | Video hidden, 100% full visual stage focus on diagram/code. | `FlowDiagram` (Flowchart), `TerminalMock` |
| **`blurred-behind`** | Dynamic GPU background blur + dimmed ambient stage. | `StrikethroughReveal`, `BulletList`, `ChatMock` |
| **`split-left`** | Speaker on Left (or Top), graphic on Right (or Bottom). | Side-by-side walkthroughs & comparisons |
| **`split-right`** | Speaker on Right (or Top), graphic on Left (or Bottom). | Side-by-side explanations & diagrams |
| **`video-top`** | Speaker in top 42%, dark stage canvas below with card. | `StatCard`, `RuleCard`, `ScreenshotFrame` |
| **`lower-third`** | Unobtrusive lower-third card over full-screen speaker. | Nameplates, captions, subtle stat callouts |
| **`full-bleed`** | Full-frame unblurred speaker with floating graphic overlay. | Full-frame talking head with floating cards |

### Supported AI Components

- **`TitleCard`**: Big hook typography with electric emphasis token (`emphasis="FAST"`).
- **`FlowDiagram`**: Flowcharts and architecture pipelines with nodes connected by vector arrows (`nodes=["CLIENT", "API", "DB"]`).
- **`StatCard`**: High-impact KPI numbers (`value="+861%"`, `label="THROUGHPUT"`).
- **`RuleCard`**: Best practice principles with struck-through anti-patterns (`struck="4.5MB LIMIT"`).
- **`StrikethroughReveal`**: Decision and verdict matrices with red strike lines and `check`/`cross` marks (✓/✗).
- **`BulletList`**: Enumerated feature or takeaway points with accent arrows (`▸`).
- **`TerminalMock`**: macOS command window with colored syntax highlighting (`$ npm run build`).
- **`ChatMock`**: Quoted direct messages or comment CTA keyword boxes (`keyword="AGENTS"`).
- **`ScreenshotFrame`**: Framed web browser or application preview with title chip.

---

## 5. Reference & Architecture

- **`transcript.json`**: AI word-level timestamps generated by CUDA Float16 Whisper.
- **`render-props.json`**: Kept spans (silence cuts), caption lines, and stage dimensions.
- **`overrides.json`**: User modifications from the Web Editor (font, color, cuts, scale).
- **`subtitles_custom.ass`**: Dynamically generated Advanced SubStation Alpha script with karaoke word animation tags (`\fscx115\fscy115\c&H...`).
- **`FFmpeg NVENC`**: `-c:v h264_nvenc -preset p4` compiles the video and subtitles at 115–200 fps.

