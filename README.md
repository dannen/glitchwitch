# glitchwitch

**glitchwitch** is a command-line video processor for generating analog-style signal corruption, cable scrambling artifacts, and extreme glitch art.

It combines spatial warping, sync failure simulation, chroma distortion, noise injection, and stateful signal behaviors.

---

## Features

- Smooth + chunked warp engines
- Stateful analog signal simulation
- RGB misalignment + chroma distortion
- Palette cycling ("color table roll")
- Vertical spike distortion
- Noise + temporal artifacts
- Fully parameterized CLI

---

## Requirements

- Python 3.9+
- OpenCV
- NumPy
- FFmpeg

Install:

    pip install opencv-python numpy

---

## Basic Usage

    python glitchwitch.py -i input.mp4 -o output.mp4

Segment:

    python glitchwitch.py -i input.mp4 -o out.mp4 --ss 00:00:10 --t 8

---

## Parameter Ranges (Key Controls)

### Warp (Smooth)

- --smooth-amp1 (0–300+): primary warp amplitude
- --smooth-amp2 (0–150): secondary warp
- --smooth-freq1 (0.002–0.02): large bends (lower = bigger waves)
- --smooth-freq2 (0.005–0.04): finer distortion
- --smooth-speed1/2 (0.01–0.5): warp motion speed
- --smooth-global-drift (0–400+): horizontal offset
- --smooth-global-drift-speed (0.01–0.6): horizontal movement speed
- --smooth-vertical-drift (0–150+): vertical offset
- --smooth-vertical-drift-speed (0.01–0.4): vertical movement speed
- --smooth-random-walk (0–1.0): randomness factor
- --smooth-random-walk-speed (0.01–0.3): randomness speed

---

### Vertical Spikes

- --vertical-spike-chance (0.0–1.0)
- --vertical-spike-strength (0–100+)
- --vertical-spike-width (0.0003–0.005)
- --vertical-spike-count-min/max (1–10)
- --vertical-spike-jitter (0.0–0.2)

---

### RGB / Color Distortion

- --rgb-shift (0–150+)
- --rgb-vertical-shift (0–20)
- --rgb-split-boost (1.0–4.0+)
- --rgb-persist-min/max (1–100 frames)
- --channel-jitter-per-line (0–10)

---

### Chroma / Hue

- --chroma-phase-oscillation (0–100+)
- --chroma-phase-speed (0.01–0.5)

---

### Palette Roll

- --palette-roll (on/off)
- --palette-roll-amount (0–180)
- --palette-roll-speed (0.01–0.3)
- --palette-roll-steps (0–32)
- --palette-roll-hold-min/max (1–60 frames)

---

### Signal Inversion

- --invert-luma-default (flag)
- --normal-flash-chance (0–0.1 typical)
- --normal-flash-min/max (1–10 frames)

---

## Recipes

### 1. Classic Scrambled Cable

    --preset smooth_scramble1     --smooth-global-drift 260     --smooth-global-drift-speed 0.08     --smooth-vertical-drift 30     --smooth-random-walk 0.25     --vertical-spike-chance 0.06     --rgb-shift 40     --chroma-phase-oscillation 20

---

### 2. Aggressive Pay-TV Scramble

    --smooth-global-drift 320     --smooth-vertical-drift 60     --smooth-random-walk 0.4     --vertical-spike-strength 80     --vertical-spike-chance 0.2     --rgb-shift 90     --rgb-split-boost 3.0     --chroma-phase-oscillation 60     --invert-luma-default

---

### 3. VHS Tracking Failure

    --warp-mode chunked     --line-wobble 10     --line-random 5     --vertical-spike-chance 0.1     --gaussian-noise 12     --ghost 0.1

---

### 4. Psychedelic Palette Drift

    --palette-roll     --palette-roll-amount 180     --palette-roll-steps 12     --palette-roll-hold-min 10     --palette-roll-hold-max 25     --chroma-phase-oscillation 70     --rgb-shift 50

---

### 5. Glitch Art Max Chaos

    --smooth-global-drift 350     --smooth-vertical-drift 80     --smooth-random-walk 0.5     --vertical-spike-strength 90     --rgb-shift 120     --channel-jitter-per-line 8     --chroma-phase-oscillation 90     --chroma-phase-speed 0.4     --palette-roll     --invert-luma-default

---

### 6. Mostly Inverted Nightmare

    --invert-luma-default     --normal-flash-chance 0.02     --rgb-shift 80     --chroma-phase-oscillation 50     --smooth-global-drift 250

---

## Randomization

    --randomize

Save:

    --save-settings settings.json

Load:

    --load-settings settings.json

---

## Notes

- Many parameters interact non-linearly
- Small changes can produce large visual differences
- Extreme settings may produce unwatchable output (intended)

---

## License

Use freely. Break video aggressively.
