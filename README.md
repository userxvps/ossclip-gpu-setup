# OSSClip GPU Acceleration & Antigravity AI Suite 🚀

Automated setup, hardware acceleration, and Cairo AI vector graphics suite for [OSSClip](https://github.com/AhsanAyaz/ossclip) on **Google Colab** or Linux with an NVIDIA GPU (**Tesla T4, RTX 30/40 series, A100, L4**, etc.).

> **Architecture Note**: This fork replaces the default Chromium/Puppeteer Remotion software rendering engine (which takes multiple minutes) with a high-performance **Cairo Vector Engine (`libcairo` + `cairosvg`)** and **NVIDIA NVENC Hardware Video Encoder (`h264_nvenc`)**, reducing render times from ~5–10 minutes down to **15–25 seconds (115–200+ FPS)** with full visual layout and subtitle parity.

---

## ⚡ Quickstart on Google Colab (Step-by-Step)

Follow these simple steps to run OSSClip with full GPU acceleration on Google Colab:

### Step 1: Select GPU Runtime
In Google Colab, go to the top menu:
> **Runtime** ➔ **Change runtime type** ➔ Select **T4 GPU** (or L4 / A100) ➔ Click **Save**.

---

### Step 2: Run the One-Click Installer
Paste and execute this command in a Colab code cell (takes ~1 minute):

```bash
!curl -fsSL https://raw.githubusercontent.com/userxvps/ossclip-gpu-setup/main/setup_ossclip_gpu.sh | bash
```

**What this automatically installs and configures:**
* ✅ **GPU Faster-Whisper** with Silero VAD, Float16 Tensor Cores, and batch-24 inference (64x realtime speech transcription).
* ✅ **Cairo Vector Graphics Engine (`libcairo` + `cairosvg`)** for pixel-perfect 4K HD graphic overlays in 12 milliseconds.
* ✅ **Tesla T4 NVENC Hardware Pipeline (`ossclip-gpu-render`)** with 32 hardware VRAM surfaces, RAM-disk buffering (`/dev/shm`), and balanced preset `p4` (38–50+ FPS).
* ✅ **Interactive Setup Wizard Patch:** Adds dedicated option to toggle AI graphics (`No` by default) and choose Cairo presets.
* ✅ **Web Editor (`ossclip edit`) Overrides Sync:** Automatically respects user cuts, retyped subtitles, hidden words, accent colors, and custom card positions (`dx`, `dy`, `scale`).
* ✅ **Creator Typography:** Installs `Montserrat`, `JetBrains Mono`, `Bebas Neue`, `Anton`, and `Rubik`.

---

### Step 3: Run the Interactive Wizard (or CLI)

#### Option A: Interactive Wizard
```bash
!ossclip produce /content/your_video.mp4
```
* Asks: `Create graphics with AI? (No = clean cut video without graphic overlays)` (Default: `No`).
* If `Yes`: Choose your Cairo AI graphics preset (`tokyo-night`, `linear`, `stripe`, `vercel`, or `auto`).

#### Option B: Direct GPU Speech Transcription (1.47s for 94s Audio!)
```bash
!ossclip transcribe /content/your_video.mp4
```

---

### Step 4: Edit in the Web Interface (`ossclip edit`)

Launch the web editor directly inside your Google Colab session:

```python
# Run this inside a Colab cell to open the Web Editor:
from google.colab.output import serve_kernel_port_as_iframe
import subprocess, threading

def start_editor():
    subprocess.run(["ossclip", "edit", "/content/your_video.mp4", "--host", "0.0.0.0", "--port", "3000"])

threading.Thread(target=start_editor, daemon=True).start()
serve_kernel_port_as_iframe(3000, width="100%", height="800")
```

**What you can do in the Web Editor:**
1. **Edit Subtitles:** Click any word in the timeline to fix spelling or retype text.
2. **Hide Words:** Hide specific words from captions without muting audio.
3. **Manual Cuts:** Use `Cmd/Ctrl+B` to cut unwanted sections.
4. **Customize Graphics & Layout:** Move cards (`dx`, `dy`), resize them (`scale`), or edit code lines and titles.
5. **Click "Render":** The web editor calls `/usr/local/bin/ossclip-gpu-render` directly—every edit is honored and exported in seconds on your GPU!

---

### Step 5: Export with Tesla T4 NVENC (`ossclip-gpu-render`)

You can also render directly from the terminal with granular controls:

```bash
# 1. Full Video with Multi-Graphic Cairo Overlays & Hormozi Captions
ossclip-gpu-render /content/your_video.mp4 --graphics-style auto -o /content/final_video.mp4

# 2. Specific Style Preset (e.g. Tokyo Night Terminal Code Window)
ossclip-gpu-render /content/your_video.mp4 --graphics-style tokyo-night -o /content/terminal_video.mp4

# 3. Clean Cut Video (100% Clean — No Graphics, No Captions)
ossclip-gpu-render /content/your_video.mp4 --no-graphics --no-captions -o /content/clean_cut.mp4

# 4. Vertical 9:16 Short / Reel with MrBeast Captions
ossclip-gpu-render /content/your_video.mp4 --format vertical --style mrbeast -o /content/vertical_reel.mp4
```

---

## 🎨 Cairo AI Vector Graphics Suite

All graphics are generated natively as vector SVGs and composited with GPU hardware acceleration in under **12 milliseconds**:

| Style | Description | Preview Frame |
|---|---|:---:|
| **Tokyo Night** | macOS Dark Window (`#16161E`), traffic light buttons, line numbers, TypeScript tokens | [View Frame](docs/benchmark/images/test_frame_02_terminal.png) |
| **Linear Obsidian** | Slate dark theme, glowing indigo/cyan borders, multi-stage cloud architecture pipeline | [View Frame](docs/benchmark/images/test_frame_01_linear.png) |
| **Stripe / Notion** | Crisp high-contrast white card (`#FFFFFF`), rejected limit panel vs best practice panel | [View Frame](docs/benchmark/images/test_frame_03_stripe.png) |
| **Vercel Geist** | Pitch-black card (`#000000`), 52pt bold numbers, electric green benchmark tags | [View Frame](docs/benchmark/images/test_frame_04_vercel.png) |

> 📊 Detailed performance benchmarks and architectural comparisons are available in [docs/benchmark/ai_graphics_gpu_benchmark.md](docs/benchmark/ai_graphics_gpu_benchmark.md).

---

## ⚡ Maximum RAM & GPU Concurrency Architecture

* **GPU Whisper ASR:** Silero VAD + batch size 24 on GPU Tensor Cores (Float16) $\rightarrow$ **1.47s execution for 94s clip (64x realtime)**.
* **RAM-Disk Buffering (`/dev/shm`):** Overlay assets and subtitle files are written to shared RAM at **> 10 GB/s** to eliminate disk latency.
* **NVENC Hardware Surfaces:** `-surfaces 32` pre-allocates 32 hardware frame surfaces in Tesla T4 VRAM so NVENC never stalls.
* **Balanced Preset `p4`:** Runs at **38–50+ FPS** on full 1080p compositions.

---

## 📁 Repository Structure

```text
├── setup_ossclip_gpu.sh      # Master one-click installation script
├── README.md                 # Documentation & Colab Quickstart guide
├── docs/
│   └── benchmark/
│       ├── ai_graphics_gpu_benchmark.md  # Detailed benchmark findings & proof
│       └── images/                       # Full 1080p sample frames
└── .agents/
    └── skills/
        └── ossclip-gpu-video-editing/
            └── SKILL.md      # Antigravity agent runbook & specs
```
