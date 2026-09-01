# OSSClip GPU Acceleration & Antigravity Suite 🚀

Automated setup and hardware-acceleration suite for [OSSClip](https://github.com/ossclip/ossclip) on Google Colab or Linux with an NVIDIA GPU (Tesla T4, RTX 30/40 series, A100, etc.).

---

## ⚡ Quickstart (1-Liner for Google Colab)

Run this single command inside a Google Colab notebook cell:

```bash
!curl -fsSL https://raw.githubusercontent.com/userxvps/ossclip-gpu-setup/main/setup_ossclip_gpu.sh | bash
```

> 📊 **GPU Benchmarks & Visual Samples:** See [AI Graphics GPU Benchmark & Visual Evaluation](docs/benchmark/ai_graphics_gpu_benchmark.md) for 100 FPS performance measurements and side-by-side screenshot comparisons.

---

## 🛠 What This Setup Includes

1. **Tesla T4 NVENC GPU Acceleration**:
   - Replaces CPU headless Chromium rendering with hardware NVENC (`h264_nvenc -preset p7`).
   - Renders 1080p videos at 120–200+ FPS (~20–35s per video).
2. **GPU Faster-Whisper (CUDA Float16)**:
   - Whisper ASR with Batched Inference on GPU.
   - ~15–18s for speech transcription with word-level timestamps.
3. **Antigravity AI Phonetic Repair (Fast Low-Latency)**:
   - Configured with `gemini-3.7-flash-low` and `--effort low` (~14s execution).
   - Injects `--dangerously-skip-permissions` for seamless child process execution.
4. **Decoupled AI Review**:
   - AI transcript review runs even when title cards and graphics are turned off.
5. **Calibrated Silence & Dead-Air Detection**:
   - Prevents background room noise (~-25 dBFS) from vetoing pause cuts.
6. **Clean Video Exporter**:
   - `ossclip-gpu-render` supports `--no-graphics` and `--no-captions` for pure cuts.
7. **Creator Typography**:
   - Automatically installs Google Fonts (`Montserrat`, `Bebas Neue`, `Anton`, `Rubik`).
   - Pre-configures developer vocabulary dictionary in `~/.ossclip/config.json`.

---

## 🎬 How to Use

### 1. Interactive CLI Wizard
```bash
ossclip
```
- Choose aspect ratio (`16:9` widescreen or `9:16` vertical).
- Choose cut level (`standard` or `aggressive`).
- Select **No** to graphics if you want clean video (Antigravity AI review will still fix mishearings).
- Check "Turn the burned-in captions off" if you want zero subtitle overlays.

### 2. Direct GPU Transcribe & Edit
```bash
ossclip transcribe /content/your_video.mp4
```

### 3. Fast GPU Render (`ossclip-gpu-render`)

- **Clean cuts (no graphics, no captions)**:
  ```bash
  ossclip-gpu-render /content/your_video.mp4 --no-graphics --no-captions -o /content/clean_edited.mp4
  ```

- **Vertical Reel with Alex Hormozi captions**:
  ```bash
  ossclip-gpu-render /content/your_video.mp4 --format vertical --style hormozi -o /content/reel.mp4
  ```

- **16:9 Widescreen with AI Scene Cards & Subtitles**:
  ```bash
  ossclip-gpu-render /content/your_video.mp4 --format original --style mrbeast -o /content/widescreen.mp4
  ```

---

## 📁 Repository Structure

```text
├── setup_ossclip_gpu.sh      # Master setup script
├── README.md                 # Documentation
├── .gitignore                # Excludes large media & temp files
└── .agents/
    └── skills/
        └── ossclip-gpu-video-editing/
            └── SKILL.md      # Antigravity agent skill runbook
```
