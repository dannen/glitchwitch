#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def run_cmd(cmd: list[str]) -> None:
    eprint("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def tool_exists(name: str) -> bool:
    return shutil.which(name) is not None

def build_default_output_paths(input_path: str) -> tuple[str, str]:
    p = Path(input_path)
    ts = datetime.now().strftime("%Y%m%d-%M%S")
    out_name = f"{p.stem}-glitch-{ts}{p.suffix}"
    json_name = f"{p.stem}-glitch-{ts}.json"
    return str(p.with_name(out_name)), str(p.with_name(json_name))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analog scramble / sync-failure / glitch-art processor")
    p.add_argument("-i", "--input", required=True)
    p.add_argument("-o", "--output", help="Output video file; defaults to inputfilename-glitch-yyyymmdd-MMSS.ext")

    p.add_argument("--ss", help="ffmpeg start time, e.g. 12.5 or 00:01:02.5")
    p.add_argument("--t", help="ffmpeg duration, e.g. 8 or 00:00:08")
    p.add_argument("--keep-temp", action="store_true")

    p.add_argument("--seed", type=int)
    p.add_argument("--randomize", action="store_true")
    p.add_argument("--save-settings")
    p.add_argument("--load-settings")
    p.add_argument(
        "--preset",
        choices=[
            "mild_bad_signal",
            "tracking_failure",
            "macrovision_nightmare",
            "total_scramble",
            "glitch_art_max",
            "cable_scramble1",
            "videocipher2",
            "smooth_scramble1",
        ],
    )

    p.add_argument("--fps", type=float)
    p.add_argument("--opencv-codec", default="mp4v")
    p.add_argument("--video-codec", default="libx264")
    p.add_argument("--crf", type=int, default=18)
    p.add_argument("--preset-encode", default="medium")
    p.add_argument("--audio-mode", choices=["copy", "aac"], default="copy")
    p.add_argument("--aac-bitrate", default="192k")

    # Base warp controls
    p.add_argument("--warp-mode", choices=["chunked", "smooth"], default="chunked")
    p.add_argument("--line-wobble", type=float, default=60.0)
    p.add_argument("--line-random", type=float, default=30.0)
    p.add_argument("--line-frequency", type=float, default=0.20)
    p.add_argument("--time-frequency", type=float, default=5.0)
    p.add_argument("--global-jitter", type=float, default=12.0)

    # Chunked warp controls
    p.add_argument("--chunk-shear-min", type=int, default=8)
    p.add_argument("--chunk-shear-max", type=int, default=48)
    p.add_argument("--chunk-shear-strength", type=float, default=120.0)
    p.add_argument("--chunk-shear-jitter", type=float, default=20.0)
    p.add_argument("--directional-bias", type=float, default=0.0,
                   help="Bias shear toward one horizontal direction; positive=right, negative=left")
    p.add_argument("--directional-bias-drift", type=float, default=0.0,
                   help="Slow drift of directional bias over time")
    p.add_argument("--chunk-persistence", type=int, default=6,
                   help="How many frames chunk offsets persist before regenerating")
    p.add_argument("--macro-shear-chance", type=float, default=0.0,
                   help="Chance of a very large slab-like shear event")
    p.add_argument("--macro-shear-strength", type=float, default=220.0)
    p.add_argument("--macro-shear-height-min", type=int, default=40)
    p.add_argument("--macro-shear-height-max", type=int, default=180)

    # Smooth warp controls
    p.add_argument("--smooth-amp1", type=float, default=80.0,
                   help="Primary smooth warp amplitude in pixels")
    p.add_argument("--smooth-amp2", type=float, default=20.0,
                   help="Secondary smooth warp amplitude in pixels")
    p.add_argument("--smooth-freq1", type=float, default=0.010,
                   help="Primary smooth vertical frequency")
    p.add_argument("--smooth-freq2", type=float, default=0.023,
                   help="Secondary smooth vertical frequency")
    p.add_argument("--smooth-speed1", type=float, default=0.40,
                   help="Primary smooth warp time speed")
    p.add_argument("--smooth-speed2", type=float, default=0.17,
                   help="Secondary smooth warp time speed")
    p.add_argument("--smooth-global-drift", type=float, default=100.0,
                   help="Whole-frame horizontal drift bias in pixels")
    p.add_argument("--smooth-global-drift-speed", type=float, default=0.10,
                   help="Whole-frame horizontal drift speed")
    p.add_argument("--smooth-random-walk", type=float, default=0.6,
                   help="Slow random walk added to smooth warp")
    p.add_argument("--smooth-random-walk-speed", type=float, default=0.08,
                   help="Speed of the slow random walk")
    p.add_argument("--smooth-field-persistence", type=int, default=90,
                   help="How long the smooth warp field persists before regeneration")
    p.add_argument("--smooth-noise", type=float, default=0.10,
                   help="Small amount of smooth field noise")
    p.add_argument("--smooth-vertical-ripple", type=float, default=0.0,
                   help="Optional very small secondary vertical ripple")
    p.add_argument("--smooth-vertical-ripple-freq", type=float, default=0.003)
    p.add_argument("--smooth-vertical-ripple-speed", type=float, default=0.12)

    # Segmenty damage
    p.add_argument("--band-shift-chance", type=float, default=0.25)
    p.add_argument("--band-min-height", type=int, default=8)
    p.add_argument("--band-max-height", type=int, default=60)
    p.add_argument("--band-max-shift", type=int, default=240)
    p.add_argument("--multi-band-max", type=int, default=3)

    # Color
    p.add_argument("--rgb-shift", type=int, default=14)
    p.add_argument("--rgb-vertical-shift", type=int, default=3)
    p.add_argument("--channel-jitter-per-line", type=float, default=0.0)
    p.add_argument("--invert-luma-always", action="store_true",
               help="Invert luma for the entire video by default")

    p.add_argument("--gain-pump-chance", type=float, default=0.18)
    p.add_argument("--gain-min", type=float, default=1.1)
    p.add_argument("--gain-max", type=float, default=2.8)
    p.add_argument("--brightness-wave", type=float, default=25.0)
    p.add_argument("--brightness-wave-speed", type=float, default=7.0)

    p.add_argument("--gaussian-noise", type=float, default=18.0)
    p.add_argument("--saltpepper", type=float, default=0.004)
    p.add_argument("--speckle-chance", type=float, default=0.15)

    p.add_argument("--ghost", type=float, default=0.20)
    p.add_argument("--frame-slip-chance", type=float, default=0.06)
    p.add_argument("--freeze-frame-chance", type=float, default=0.03)
    p.add_argument("--freeze-min", type=int, default=2)
    p.add_argument("--freeze-max", type=int, default=8)

    p.add_argument("--tracking-bar-chance", type=float, default=0.18)
    p.add_argument("--tracking-bar-height", type=int, default=28)
    p.add_argument("--tracking-bar-shift", type=int, default=180)
    p.add_argument("--bottom-tear-chance", type=float, default=0.15)

    p.add_argument("--dropout-chance", type=float, default=0.08)
    p.add_argument("--dropout-lines-min", type=int, default=2)
    p.add_argument("--dropout-lines-max", type=int, default=20)
    p.add_argument("--dropout-width-min", type=int, default=50)
    p.add_argument("--dropout-width-max", type=int, default=600)
    p.add_argument("--blank-frame-chance", type=float, default=0.01)
    p.add_argument("--desat-burst-chance", type=float, default=0.08)
    
    p.add_argument("--rgb-persist-min", type=int, default=1,
               help="Minimum frames to hold RGB split state")
    p.add_argument("--rgb-persist-max", type=int, default=1,
               help="Maximum frames to hold RGB split state")
    p.add_argument("--rgb-split-boost", type=float, default=1.0,
               help="Multiplier for persistent RGB channel offsets")

    p.add_argument("--max-width", type=int)
    p.add_argument("--max-height", type=int)

    # Stateful cable / VCII-ish controls
    p.add_argument("--lock-probability", type=float, default=0.0)
    p.add_argument("--lock-strength", type=float, default=0.35)
    p.add_argument("--drift-band-count", type=int, default=0)
    p.add_argument("--drift-band-height-min", type=int, default=18)
    p.add_argument("--drift-band-height-max", type=int, default=90)
    p.add_argument("--drift-band-strength", type=float, default=1.6)
    p.add_argument("--drift-band-vertical-speed", type=float, default=0.7)
    p.add_argument("--drift-band-offset-speed", type=float, default=0.15)
    p.add_argument("--band-brightness-coupling", type=float, default=0.0)
    p.add_argument("--band-darken", type=float, default=0.0)
    p.add_argument("--chroma-phase-oscillation", type=float, default=0.0)
    p.add_argument("--chroma-phase-speed", type=float, default=1.8)
    p.add_argument("--invert-luma-chance", type=float, default=0.0)
    p.add_argument("--partial-invert-band-chance", type=float, default=0.0)
    p.add_argument("--hbi-corruption-chance", type=float, default=0.0)
    p.add_argument("--hbi-height", type=int, default=18)
    p.add_argument("--sync-loss-blanking", type=float, default=0.0)
    p.add_argument("--monochrome-chance", type=float, default=0.0)
    
    p.add_argument("--smooth-vertical-drift", type=float, default=0.0,
               help="Whole-frame vertical drift bias in pixels")
    p.add_argument("--smooth-vertical-drift-speed", type=float, default=0.08,
               help="Whole-frame vertical drift speed")

    p.add_argument("--vertical-spike-chance", type=float, default=0.0,
               help="Chance of a random vertical spike event starting")
    p.add_argument("--vertical-spike-min", type=int, default=8,
               help="Minimum number of frames a vertical spike event lasts")
    p.add_argument("--vertical-spike-max", type=int, default=28,
               help="Maximum number of frames a vertical spike event lasts")
    p.add_argument("--vertical-spike-strength", type=float, default=18.0,
               help="Strength of added spike distortion in pixels")
    p.add_argument("--vertical-spike-width", type=float, default=0.0018,
               help="Width of spike lobes across the frame height")
    p.add_argument("--vertical-spike-count-min", type=int, default=1,
               help="Minimum number of spike centers per event")
    p.add_argument("--vertical-spike-count-max", type=int, default=4,
               help="Maximum number of spike centers per event")
    p.add_argument("--vertical-spike-jitter", type=float, default=0.15,
               help="Per-frame movement/jitter of spike centers")
    p.add_argument("--invert-luma-default", action="store_true",
               help="Keep luma inverted by default")
    p.add_argument("--normal-flash-chance", type=float, default=0.0,
               help="Chance of briefly flashing back to normal (non-inverted) luma")
    p.add_argument("--normal-flash-min", type=int, default=2,
               help="Minimum frames for a normal-color flash")
    p.add_argument("--normal-flash-max", type=int, default=8,
               help="Maximum frames for a normal-color flash")
    
    p.add_argument("--palette-roll", action="store_true",
               help="Enable smooth cyclic palette shifting")
    p.add_argument("--palette-roll-amount", type=float, default=0.0,
               help="Maximum hue rotation amount in HSV hue units (0-180)")
    p.add_argument("--palette-roll-speed", type=float, default=0.05,
               help="Speed of continuous palette rotation")
    p.add_argument("--palette-roll-steps", type=int, default=0,
               help="If >0, quantize the hue rotation into this many steps around the hue wheel")
    p.add_argument("--palette-roll-hold-min", type=int, default=1,
               help="Minimum frames to hold a quantized palette step")
    p.add_argument("--palette-roll-hold-max", type=int, default=1,
               help="Maximum frames to hold a quantized palette step")
    p.add_argument("--palette-roll-saturation-scale", type=float, default=1.0,
               help="Optional saturation scaling during palette roll")
    p.add_argument("--palette-roll-value-scale", type=float, default=1.0,
               help="Optional value scaling during palette roll")

    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


@dataclass
class Settings:
    input: str
    output: str

    ss: Optional[str] = None
    t: Optional[str] = None
    keep_temp: bool = False
    invert_luma_always: bool = False
    rgb_persist_min: int = 1
    rgb_persist_max: int = 1
    rgb_split_boost: float = 1.0

    seed: Optional[int] = None
    randomize: bool = False
    save_settings: Optional[str] = None
    load_settings: Optional[str] = None
    preset: Optional[str] = None

    fps: Optional[float] = None
    opencv_codec: str = "mp4v"
    video_codec: str = "libx264"
    crf: int = 18
    preset_encode: str = "medium"
    audio_mode: str = "copy"
    aac_bitrate: str = "192k"

    warp_mode: str = "chunked"
    line_wobble: float = 60.0
    line_random: float = 30.0
    line_frequency: float = 0.20
    time_frequency: float = 5.0
    global_jitter: float = 12.0

    chunk_shear_min: int = 8
    chunk_shear_max: int = 48
    chunk_shear_strength: float = 120.0
    chunk_shear_jitter: float = 20.0
    directional_bias: float = 0.0
    directional_bias_drift: float = 0.0
    chunk_persistence: int = 6
    macro_shear_chance: float = 0.0
    macro_shear_strength: float = 220.0
    macro_shear_height_min: int = 40
    macro_shear_height_max: int = 180

    smooth_amp1: float = 80.0
    smooth_amp2: float = 20.0
    smooth_freq1: float = 0.010
    smooth_freq2: float = 0.023
    smooth_speed1: float = 0.40
    smooth_speed2: float = 0.17
    smooth_global_drift: float = 100.0
    smooth_global_drift_speed: float = 0.10
    smooth_random_walk: float = 0.6
    smooth_random_walk_speed: float = 0.08
    smooth_field_persistence: int = 90
    smooth_noise: float = 0.10
    smooth_vertical_ripple: float = 0.0
    smooth_vertical_ripple_freq: float = 0.003
    smooth_vertical_ripple_speed: float = 0.12
    smooth_vertical_drift: float = 0.0
    smooth_vertical_drift_speed: float = 0.08

    vertical_spike_chance: float = 0.0
    vertical_spike_min: int = 8
    vertical_spike_max: int = 28
    vertical_spike_strength: float = 18.0
    vertical_spike_width: float = 0.0018
    vertical_spike_count_min: int = 1
    vertical_spike_count_max: int = 4
    vertical_spike_jitter: float = 0.15

    band_shift_chance: float = 0.25
    band_min_height: int = 8
    band_max_height: int = 60
    band_max_shift: int = 240
    multi_band_max: int = 3

    rgb_shift: int = 14
    rgb_vertical_shift: int = 3
    channel_jitter_per_line: float = 0.0

    gain_pump_chance: float = 0.18
    gain_min: float = 1.1
    gain_max: float = 2.8
    brightness_wave: float = 25.0
    brightness_wave_speed: float = 7.0

    gaussian_noise: float = 18.0
    saltpepper: float = 0.004
    speckle_chance: float = 0.15

    ghost: float = 0.20
    frame_slip_chance: float = 0.06
    freeze_frame_chance: float = 0.03
    freeze_min: int = 2
    freeze_max: int = 8

    tracking_bar_chance: float = 0.18
    tracking_bar_height: int = 28
    tracking_bar_shift: int = 180
    bottom_tear_chance: float = 0.15

    dropout_chance: float = 0.08
    dropout_lines_min: int = 2
    dropout_lines_max: int = 20
    dropout_width_min: int = 50
    dropout_width_max: int = 600
    blank_frame_chance: float = 0.01
    desat_burst_chance: float = 0.08

    max_width: Optional[int] = None
    max_height: Optional[int] = None

    lock_probability: float = 0.0
    lock_strength: float = 0.35
    drift_band_count: int = 0
    drift_band_height_min: int = 18
    drift_band_height_max: int = 90
    drift_band_strength: float = 1.6
    drift_band_vertical_speed: float = 0.7
    drift_band_offset_speed: float = 0.15
    band_brightness_coupling: float = 0.0
    band_darken: float = 0.0
    chroma_phase_oscillation: float = 0.0
    chroma_phase_speed: float = 1.8
    invert_luma_chance: float = 0.0
    partial_invert_band_chance: float = 0.0
    hbi_corruption_chance: float = 0.0
    hbi_height: int = 18
    sync_loss_blanking: float = 0.0
    monochrome_chance: float = 0.0
    
    invert_luma_default: bool = False
    normal_flash_chance: float = 0.0
    normal_flash_min: int = 2
    normal_flash_max: int = 8

    verbose: bool = False

    effective_input: Optional[str] = None
    processing_width: Optional[int] = None
    processing_height: Optional[int] = None
    sample_has_audio: Optional[bool] = None
    
    palette_roll: bool = False
    palette_roll_amount: float = 0.0
    palette_roll_speed: float = 0.05
    palette_roll_steps: int = 0
    palette_roll_hold_min: int = 1
    palette_roll_hold_max: int = 1
    palette_roll_saturation_scale: float = 1.0
    palette_roll_value_scale: float = 1.0


def args_to_settings(args: argparse.Namespace) -> Settings:
    return Settings(**vars(args))


def load_settings_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_loaded_settings(base: Settings, loaded: Dict[str, Any]) -> Settings:
    current = asdict(base)
    for k, v in loaded.items():
        if k in current and k not in {"input", "output", "save_settings", "load_settings"}:
            current[k] = v
    current["input"] = base.input
    current["output"] = base.output
    current["save_settings"] = base.save_settings
    current["load_settings"] = base.load_settings
    return Settings(**current)


def apply_preset(s: Settings, name: Optional[str]) -> None:
    if not name:
        return

    presets = {
        "mild_bad_signal": dict(
            warp_mode="chunked",
            line_wobble=30, line_random=12, global_jitter=6,
            band_shift_chance=0.15, band_max_shift=120, multi_band_max=2,
            rgb_shift=8, gain_pump_chance=0.08, gaussian_noise=10,
            ghost=0.10, tracking_bar_chance=0.08, dropout_chance=0.03
        ),
        "tracking_failure": dict(
            warp_mode="chunked",
            line_wobble=100, line_random=45, global_jitter=18,
            band_shift_chance=0.45, band_max_shift=320, multi_band_max=5,
            rgb_shift=18, channel_jitter_per_line=4,
            gain_pump_chance=0.22, gaussian_noise=20,
            ghost=0.25, tracking_bar_chance=0.35, bottom_tear_chance=0.35,
            dropout_chance=0.10
        ),
        "macrovision_nightmare": dict(
            warp_mode="chunked",
            line_wobble=125, line_random=50, global_jitter=20,
            band_shift_chance=0.55, band_max_shift=420, multi_band_max=6,
            rgb_shift=22, channel_jitter_per_line=6,
            gain_pump_chance=0.35, gain_max=3.8,
            brightness_wave=40, gaussian_noise=22,
            ghost=0.30, tracking_bar_chance=0.40, bottom_tear_chance=0.30,
            dropout_chance=0.16, desat_burst_chance=0.14
        ),
        "total_scramble": dict(
            warp_mode="chunked",
            line_wobble=150, line_random=70, global_jitter=28,
            band_shift_chance=0.70, band_max_shift=520, multi_band_max=8,
            rgb_shift=30, rgb_vertical_shift=5, channel_jitter_per_line=8,
            gain_pump_chance=0.45, gain_max=4.2,
            brightness_wave=55, gaussian_noise=30, saltpepper=0.010,
            ghost=0.40, frame_slip_chance=0.16,
            tracking_bar_chance=0.45, bottom_tear_chance=0.45,
            dropout_chance=0.24, blank_frame_chance=0.03,
            desat_burst_chance=0.20
        ),
        "glitch_art_max": dict(
            warp_mode="chunked",
            line_wobble=220, line_random=100, global_jitter=45,
            band_shift_chance=0.85, band_max_shift=800, multi_band_max=10,
            rgb_shift=42, rgb_vertical_shift=8, channel_jitter_per_line=12,
            gain_pump_chance=0.60, gain_max=5.0,
            brightness_wave=80, brightness_wave_speed=12,
            gaussian_noise=40, saltpepper=0.018, speckle_chance=0.45,
            ghost=0.55, frame_slip_chance=0.24, freeze_frame_chance=0.10,
            tracking_bar_chance=0.60, bottom_tear_chance=0.50,
            dropout_chance=0.32, blank_frame_chance=0.06,
            desat_burst_chance=0.30
        ),
        "cable_scramble1": dict(
            warp_mode="chunked",
            line_wobble=24,
            line_random=6,
            global_jitter=4,
            line_frequency=0.06,
            time_frequency=1.2,

            band_shift_chance=0.03,
            band_max_shift=60,
            multi_band_max=1,

            rgb_shift=8,
            rgb_vertical_shift=1,
            channel_jitter_per_line=1.2,

            gain_pump_chance=0.05,
            gain_max=1.35,
            brightness_wave=4,
            gaussian_noise=8,
            saltpepper=0.0008,

            ghost=0.03,
            frame_slip_chance=0.01,

            tracking_bar_chance=0.05,
            bottom_tear_chance=0.04,
            dropout_chance=0.0,
            blank_frame_chance=0.0,
            desat_burst_chance=0.03,

            lock_probability=0.18,
            lock_strength=0.55,

            drift_band_count=2,
            drift_band_height_min=40,
            drift_band_height_max=120,
            drift_band_strength=1.5,
            drift_band_vertical_speed=0.25,
            drift_band_offset_speed=0.05,

            band_brightness_coupling=10.0,
            band_darken=0.10,

            chroma_phase_oscillation=3.0,
            chroma_phase_speed=0.8,

            hbi_corruption_chance=0.08,
            hbi_height=14,
            sync_loss_blanking=0.04,
            monochrome_chance=0.01,

            chunk_shear_min=18,
            chunk_shear_max=72,
            chunk_shear_strength=150.0,
            chunk_shear_jitter=8.0,
            directional_bias=95.0,
            directional_bias_drift=0.25,
            chunk_persistence=10,
            macro_shear_chance=0.08,
            macro_shear_strength=260.0,
            macro_shear_height_min=70,
            macro_shear_height_max=180,
        ),
        "videocipher2": dict(
            warp_mode="chunked",
            line_wobble=92, line_random=20, global_jitter=14,
            line_frequency=0.10, time_frequency=2.0,
            band_shift_chance=0.12, band_max_shift=120, multi_band_max=2,
            rgb_shift=4, rgb_vertical_shift=0, channel_jitter_per_line=1.0,
            gain_pump_chance=0.08, gain_min=1.0, gain_max=1.4,
            brightness_wave=6, brightness_wave_speed=2.0,
            gaussian_noise=8, saltpepper=0.0005, speckle_chance=0.04,
            ghost=0.02, frame_slip_chance=0.01, freeze_frame_chance=0.0,
            tracking_bar_chance=0.08, bottom_tear_chance=0.06,
            dropout_chance=0.01, blank_frame_chance=0.0,
            desat_burst_chance=0.02,
            lock_probability=0.20, lock_strength=0.55,
            drift_band_count=2,
            drift_band_height_min=20, drift_band_height_max=60,
            drift_band_strength=2.1,
            drift_band_vertical_speed=0.4,
            drift_band_offset_speed=0.08,
            band_brightness_coupling=12.0,
            band_darken=0.08,
            chroma_phase_oscillation=2.0,
            chroma_phase_speed=0.9,
            invert_luma_chance=0.18,
            partial_invert_band_chance=0.22,
            hbi_corruption_chance=0.38,
            hbi_height=22,
            sync_loss_blanking=0.22,
            monochrome_chance=0.10,
            chunk_shear_min=16,
            chunk_shear_max=52,
            chunk_shear_strength=70.0,
            chunk_shear_jitter=5.0,
            directional_bias=20.0,
            directional_bias_drift=0.1,
            chunk_persistence=8,
            macro_shear_chance=0.02,
            macro_shear_strength=120.0,
            macro_shear_height_min=40,
            macro_shear_height_max=120,
        ),
        "smooth_scramble1": dict(
            warp_mode="smooth",
            line_wobble=0.0,
            line_random=0.10,
            global_jitter=0.5,

            smooth_amp1=62.0,
            smooth_amp2=11.0,
            smooth_freq1=0.010,
            smooth_freq2=0.021,
            smooth_speed1=0.28,
            smooth_speed2=0.11,
            smooth_global_drift=95.0,
            smooth_global_drift_speed=0.08,
            smooth_random_walk=0.35,
            smooth_random_walk_speed=0.05,
            smooth_field_persistence=140,
            smooth_noise=0.03,
            smooth_vertical_ripple=1.5,
            smooth_vertical_ripple_freq=0.002,
            smooth_vertical_ripple_speed=0.08,

            band_shift_chance=0.0,
            multi_band_max=0,
            drift_band_count=0,
            tracking_bar_chance=0.0,
            bottom_tear_chance=0.0,
            dropout_chance=0.0,
            blank_frame_chance=0.0,

            rgb_shift=18,
            rgb_vertical_shift=1,
            channel_jitter_per_line=0.0,
            chroma_phase_oscillation=10.0,
            chroma_phase_speed=0.45,

            invert_luma_chance=0.16,
            partial_invert_band_chance=0.02,
            monochrome_chance=0.0,

            gain_pump_chance=0.01,
            gain_min=1.0,
            gain_max=1.12,
            brightness_wave=2.0,
            brightness_wave_speed=0.22,

            gaussian_noise=8.0,
            saltpepper=0.0006,
            ghost=0.01,
        ),
    }

    for k, v in presets[name].items():
        setattr(s, k, v)


def randomize_settings(s: Settings, rng: random.Random) -> None:
    s.line_wobble = rng.uniform(40, 220)
    s.line_random = rng.uniform(10, 100)
    s.line_frequency = rng.uniform(0.05, 0.45)
    s.time_frequency = rng.uniform(2.0, 14.0)
    s.global_jitter = rng.uniform(4, 45)

    s.band_shift_chance = rng.uniform(0.15, 0.90)
    s.band_min_height = rng.randint(4, 20)
    s.band_max_height = rng.randint(30, 120)
    s.band_max_shift = rng.randint(120, 900)
    s.multi_band_max = rng.randint(1, 10)

    s.rgb_shift = rng.randint(4, 45)
    s.rgb_vertical_shift = rng.randint(0, 8)
    s.channel_jitter_per_line = rng.uniform(0, 12)

    s.gain_pump_chance = rng.uniform(0.05, 0.65)
    s.gain_min = rng.uniform(1.05, 1.40)
    s.gain_max = rng.uniform(1.8, 5.0)
    s.brightness_wave = rng.uniform(0, 90)
    s.brightness_wave_speed = rng.uniform(1.0, 20.0)

    s.gaussian_noise = rng.uniform(4, 45)
    s.saltpepper = rng.uniform(0.0, 0.02)
    s.speckle_chance = rng.uniform(0.0, 0.50)

    s.ghost = rng.uniform(0.0, 0.65)
    s.frame_slip_chance = rng.uniform(0.0, 0.30)
    s.freeze_frame_chance = rng.uniform(0.0, 0.15)
    s.freeze_min = rng.randint(2, 5)
    s.freeze_max = rng.randint(6, 18)

    s.tracking_bar_chance = rng.uniform(0.0, 0.65)
    s.tracking_bar_height = rng.randint(8, 80)
    s.tracking_bar_shift = rng.randint(40, 500)
    s.bottom_tear_chance = rng.uniform(0.0, 0.55)

    s.dropout_chance = rng.uniform(0.0, 0.35)
    s.dropout_lines_min = rng.randint(1, 4)
    s.dropout_lines_max = rng.randint(6, 50)
    s.dropout_width_min = rng.randint(10, 200)
    s.dropout_width_max = rng.randint(200, 1200)
    s.blank_frame_chance = rng.uniform(0.0, 0.08)
    s.desat_burst_chance = rng.uniform(0.0, 0.35)

    s.lock_probability = rng.uniform(0.0, 0.40)
    s.lock_strength = rng.uniform(0.20, 0.80)
    s.drift_band_count = rng.randint(0, 5)
    s.drift_band_height_min = rng.randint(12, 40)
    s.drift_band_height_max = rng.randint(40, 120)
    s.drift_band_strength = rng.uniform(1.0, 4.0)
    s.drift_band_vertical_speed = rng.uniform(0.1, 1.4)
    s.drift_band_offset_speed = rng.uniform(0.05, 0.35)
    s.band_brightness_coupling = rng.uniform(0.0, 40.0)
    s.band_darken = rng.uniform(0.0, 0.35)
    s.chroma_phase_oscillation = rng.uniform(0.0, 12.0)
    s.chroma_phase_speed = rng.uniform(0.3, 3.0)
    s.invert_luma_chance = rng.uniform(0.0, 0.25)
    s.partial_invert_band_chance = rng.uniform(0.0, 0.30)
    s.hbi_corruption_chance = rng.uniform(0.0, 0.45)
    s.hbi_height = rng.randint(10, 30)
    s.sync_loss_blanking = rng.uniform(0.0, 0.35)
    s.monochrome_chance = rng.uniform(0.0, 0.15)

    s.chunk_shear_min = rng.randint(8, 32)
    s.chunk_shear_max = rng.randint(max(s.chunk_shear_min + 4, 24), 120)
    s.chunk_shear_strength = rng.uniform(40.0, 260.0)
    s.chunk_shear_jitter = rng.uniform(0.0, 30.0)
    s.directional_bias = rng.uniform(-180.0, 180.0)
    s.directional_bias_drift = rng.uniform(0.0, 1.0)
    s.chunk_persistence = rng.randint(2, 18)
    s.macro_shear_chance = rng.uniform(0.0, 0.20)
    s.macro_shear_strength = rng.uniform(80.0, 420.0)
    s.macro_shear_height_min = rng.randint(20, 100)
    s.macro_shear_height_max = rng.randint(max(s.macro_shear_height_min + 10, 60), 260)

    s.smooth_amp1 = rng.uniform(20.0, 140.0)
    s.smooth_amp2 = rng.uniform(0.0, 35.0)
    s.smooth_freq1 = rng.uniform(0.003, 0.025)
    s.smooth_freq2 = rng.uniform(0.008, 0.050)
    s.smooth_speed1 = rng.uniform(0.03, 1.0)
    s.smooth_speed2 = rng.uniform(0.03, 0.5)
    s.smooth_global_drift = rng.uniform(-180.0, 180.0)
    s.smooth_global_drift_speed = rng.uniform(0.01, 0.4)
    s.smooth_random_walk = rng.uniform(0.0, 2.0)
    s.smooth_random_walk_speed = rng.uniform(0.01, 0.2)
    s.smooth_field_persistence = rng.randint(20, 220)
    s.smooth_noise = rng.uniform(0.0, 0.2)
    s.smooth_vertical_ripple = rng.uniform(0.0, 5.0)
    s.smooth_vertical_ripple_freq = rng.uniform(0.001, 0.01)
    s.smooth_vertical_ripple_speed = rng.uniform(0.01, 0.25)


def save_settings_json(path: str, settings: Settings) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(settings), f, indent=2, sort_keys=True)


def maybe_extract_sample(settings: Settings) -> tuple[str, Optional[str]]:
    if not settings.ss and not settings.t:
        return settings.input, None

    suffix = Path(settings.input).suffix or ".mp4"
    fd, temp_path = tempfile.mkstemp(prefix="glitch_sample_", suffix=suffix)
    os.close(fd)

    cmd = ["ffmpeg", "-y"]
    if settings.ss:
        cmd += ["-ss", str(settings.ss)]
    cmd += ["-i", settings.input]
    if settings.t:
        cmd += ["-t", str(settings.t)]
    cmd += [
        "-map", "0:v:0?",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-c:a", "aac",
        "-b:a", "192k",
        "-pix_fmt", "yuv420p",
        temp_path,
    ]
    run_cmd(cmd)
    return temp_path, temp_path


def sample_has_audio(path: str) -> bool:
    if not tool_exists("ffprobe"):
        return True
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=index",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return bool(res.stdout.strip())


def resize_if_needed(frame: np.ndarray, settings: Settings) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = 1.0
    if settings.max_width and w > settings.max_width:
        scale = min(scale, settings.max_width / w)
    if settings.max_height and h > settings.max_height:
        scale = min(scale, settings.max_height / h)
    if scale < 1.0:
        frame = cv2.resize(
            frame,
            (max(2, int(w * scale)), max(2, int(h * scale))),
            interpolation=cv2.INTER_AREA
        )
    return frame


def clip_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0, 255).astype(np.uint8)


def add_gaussian_noise(frame: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    if sigma <= 0:
        return frame
    noise = rng.normal(0, sigma, frame.shape).astype(np.float32)
    return clip_uint8(frame.astype(np.float32) + noise)


def add_salt_pepper(frame: np.ndarray, prob: float, rng: np.random.Generator) -> np.ndarray:
    if prob <= 0:
        return frame
    out = frame.copy()
    h, w = out.shape[:2]
    n = int(h * w * prob)
    if n <= 0:
        return out
    ys = rng.integers(0, h, size=n)
    xs = rng.integers(0, w, size=n)
    vals = rng.integers(0, 2, size=n) * 255
    out[ys, xs] = np.stack([vals, vals, vals], axis=1)
    return out


def roll_frame(frame: np.ndarray, shift_x: int = 0, shift_y: int = 0) -> np.ndarray:
    out = frame
    if shift_x:
        out = np.roll(out, shift_x, axis=1)
    if shift_y:
        out = np.roll(out, shift_y, axis=0)
    return out


def init_state(settings: Settings, height: int, width: int, py_rng: random.Random) -> Dict[str, Any]:
    drift_bands = []
    for _ in range(settings.drift_band_count):
        band_h = py_rng.randint(
            settings.drift_band_height_min,
            max(settings.drift_band_height_min, settings.drift_band_height_max)
        )
        y = py_rng.uniform(0, max(1, height - band_h))
        drift_bands.append({
            "y": y,
            "height": band_h,
            "phase": py_rng.uniform(0, math.pi * 2.0),
            "offset_phase": py_rng.uniform(0, math.pi * 2.0),
            "direction": py_rng.choice([-1.0, 1.0]),
        })

    return {
        "drift_bands": drift_bands,
        "lock_state": False,
        "lock_countdown": 0,
        "invert_countdown": 0,
        "mono_countdown": 0,
        "sync_loss_countdown": 0,
        "chunk_shear": [],
        "chunk_shear_countdown": 0,
        "smooth_phase1": py_rng.uniform(0.0, math.pi * 2.0),
        "smooth_phase2": py_rng.uniform(0.0, math.pi * 2.0),
        "smooth_walk_value": py_rng.uniform(-1.0, 1.0),
        "smooth_walk_target": py_rng.uniform(-1.0, 1.0),
        "smooth_field_countdown": 0,
        "smooth_noise_phase": py_rng.uniform(0.0, math.pi * 2.0),
        "rgb_countdown": 0,
        "rgb_gx": 0,
        "rgb_rx": 0,
        "rgb_gy": 0,
        "rgb_ry": 0,
        "vertical_spike_countdown": 0,
        "vertical_spike_centers": [],
        "vertical_spike_signs": [],
        "normal_flash_countdown": 0,
        "palette_roll_step_index": 0,
        "palette_roll_countdown": 0,
    }


def update_state(state: Dict[str, Any], settings: Settings, frame_idx: int, height: int, py_rng: random.Random) -> None:
    for band in state["drift_bands"]:
        band["y"] += band["direction"] * settings.drift_band_vertical_speed
        if band["y"] < 0:
            band["y"] = 0
            band["direction"] *= -1
        if band["y"] > max(0, height - band["height"]):
            band["y"] = max(0, height - band["height"])
            band["direction"] *= -1
        band["phase"] += settings.drift_band_vertical_speed * 0.01
        band["offset_phase"] += settings.drift_band_offset_speed

    if state["lock_countdown"] <= 0 and py_rng.random() < settings.lock_probability:
        state["lock_state"] = True
        state["lock_countdown"] = py_rng.randint(4, 18)
    elif state["lock_countdown"] > 0:
        state["lock_countdown"] -= 1
        if state["lock_countdown"] <= 0:
            state["lock_state"] = False

    if state["invert_countdown"] <= 0 and py_rng.random() < settings.invert_luma_chance:
        state["invert_countdown"] = py_rng.randint(2, 8)
    elif state["invert_countdown"] > 0:
        state["invert_countdown"] -= 1

    if state["mono_countdown"] <= 0 and py_rng.random() < settings.monochrome_chance:
        state["mono_countdown"] = py_rng.randint(2, 10)
    elif state["mono_countdown"] > 0:
        state["mono_countdown"] -= 1

    if state["sync_loss_countdown"] <= 0 and py_rng.random() < settings.sync_loss_blanking:
        state["sync_loss_countdown"] = py_rng.randint(1, 6)
    elif state["sync_loss_countdown"] > 0:
        state["sync_loss_countdown"] -= 1

    # Slow random walk for smooth mode
    if state["smooth_field_countdown"] <= 0:
        state["smooth_walk_target"] = py_rng.uniform(-1.0, 1.0)
        state["smooth_field_countdown"] = settings.smooth_field_persistence
    else:
        state["smooth_field_countdown"] -= 1

    delta = state["smooth_walk_target"] - state["smooth_walk_value"]
    state["smooth_walk_value"] += delta * max(0.001, settings.smooth_random_walk_speed)
    state["smooth_phase1"] += settings.smooth_speed1 * 0.03
    state["smooth_phase2"] += settings.smooth_speed2 * 0.03
    state["smooth_noise_phase"] += 0.013
    
    if state["vertical_spike_countdown"] <= 0:
        if py_rng.random() < settings.vertical_spike_chance:
            count = py_rng.randint(settings.vertical_spike_count_min, settings.vertical_spike_count_max)
            state["vertical_spike_centers"] = [py_rng.uniform(0.0, 1.0) for _ in range(count)]
            state["vertical_spike_signs"] = [py_rng.choice([-1.0, 1.0]) for _ in range(count)]
            state["vertical_spike_countdown"] = py_rng.randint(settings.vertical_spike_min, settings.vertical_spike_max)
    else:
        state["vertical_spike_countdown"] -= 1
        new_centers = []
        for c in state["vertical_spike_centers"]:
            c += py_rng.uniform(-settings.vertical_spike_jitter, settings.vertical_spike_jitter) * 0.01
            c = max(0.0, min(1.0, c))
            new_centers.append(c)
        state["vertical_spike_centers"] = new_centers
        
    if state["normal_flash_countdown"] <= 0:
        if settings.normal_flash_chance > 0 and py_rng.random() < settings.normal_flash_chance:
            state["normal_flash_countdown"] = py_rng.randint(
                settings.normal_flash_min,
                settings.normal_flash_max
            )
    else:
        state["normal_flash_countdown"] -= 1
        
def apply_palette_roll(
    frame: np.ndarray,
    settings: Settings,
    py_rng: random.Random,
    frame_idx: int,
    state: Dict[str, Any]
) -> np.ndarray:
    if not settings.palette_roll or settings.palette_roll_amount == 0:
        return frame

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)

    if settings.palette_roll_steps and settings.palette_roll_steps > 0:
        if state["palette_roll_countdown"] <= 0:
            state["palette_roll_step_index"] = (state["palette_roll_step_index"] + 1) % settings.palette_roll_steps
            state["palette_roll_countdown"] = py_rng.randint(
                settings.palette_roll_hold_min,
                settings.palette_roll_hold_max
            )
        else:
            state["palette_roll_countdown"] -= 1

        step_size = settings.palette_roll_amount / settings.palette_roll_steps
        hue_shift = state["palette_roll_step_index"] * step_size
    else:
        hue_shift = (math.sin(frame_idx * 0.03 * settings.palette_roll_speed) * 0.5 + 0.5) * settings.palette_roll_amount

    hsv[..., 0] = (hsv[..., 0] + hue_shift) % 180.0
    hsv[..., 1] *= settings.palette_roll_saturation_scale
    hsv[..., 2] *= settings.palette_roll_value_scale

    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2], 0, 255)

    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def ensure_chunk_state(state: Dict[str, Any], settings: Settings, h: int, py_rng: random.Random) -> None:
    regen = False
    if "chunk_shear" not in state:
        regen = True
    elif state.get("chunk_shear_countdown", 0) <= 0:
        regen = True

    if not regen:
        state["chunk_shear_countdown"] -= 1
        return

    chunks = []
    y = 0
    while y < h:
        ch = py_rng.randint(settings.chunk_shear_min, settings.chunk_shear_max)
        ch = min(ch, h - y)
        chunks.append({
            "y0": y,
            "y1": y + ch,
            "offset": py_rng.uniform(-settings.chunk_shear_strength, settings.chunk_shear_strength),
            "tilt": py_rng.uniform(-0.35, 0.35),
        })
        y += ch

    state["chunk_shear"] = chunks
    state["chunk_shear_countdown"] = settings.chunk_persistence


def apply_chunked_warp(
    frame: np.ndarray,
    frame_idx: int,
    settings: Settings,
    py_rng: random.Random,
    state: Dict[str, Any]
) -> np.ndarray:
    h, _ = frame.shape[:2]
    out = np.empty_like(frame)

    ensure_chunk_state(state, settings, h, py_rng)

    wobble_scale = 1.0
    rand_scale = 1.0
    if state["lock_state"]:
        wobble_scale = max(0.0, 1.0 - settings.lock_strength)
        rand_scale = max(0.0, 1.0 - settings.lock_strength * 0.85)

    bias = settings.directional_bias
    if settings.directional_bias_drift != 0:
        bias += math.sin(frame_idx * 0.03 * settings.directional_bias_drift) * abs(settings.directional_bias)

    global_shift = py_rng.uniform(-settings.global_jitter, settings.global_jitter) * rand_scale

    for y in range(h):
        sinusoid = math.sin(y * settings.line_frequency + frame_idx * settings.time_frequency * 0.03)
        shift = sinusoid * settings.line_wobble * wobble_scale
        shift += py_rng.uniform(-settings.line_random, settings.line_random) * rand_scale
        shift += global_shift
        shift += bias

        for chunk in state["chunk_shear"]:
            if chunk["y0"] <= y < chunk["y1"]:
                rel = (y - chunk["y0"]) / max(1, (chunk["y1"] - chunk["y0"]))
                chunk_shift = chunk["offset"] + ((rel - 0.5) * chunk["tilt"] * settings.chunk_shear_strength)
                chunk_shift += py_rng.uniform(-settings.chunk_shear_jitter, settings.chunk_shear_jitter)
                shift += chunk_shift
                break

        for band in state["drift_bands"]:
            y0 = int(band["y"])
            y1 = min(h, y0 + band["height"])
            if y0 <= y < y1:
                rel = (y - y0) / max(1, band["height"])
                envelope = 0.5 - 0.5 * math.cos(rel * math.pi)
                band_shift = math.sin(band["offset_phase"] + rel * math.pi * 1.4)
                shift += band_shift * settings.line_wobble * settings.drift_band_strength * envelope

        out[y] = np.roll(frame[y], int(shift), axis=0)

    if py_rng.random() < settings.macro_shear_chance:
        bh = py_rng.randint(settings.macro_shear_height_min, settings.macro_shear_height_max)
        bh = min(bh, h)
        y0 = py_rng.randint(0, max(0, h - bh))
        slab_shift = int(py_rng.uniform(
            settings.directional_bias - settings.macro_shear_strength,
            settings.directional_bias + settings.macro_shear_strength
        ))
        out[y0:y0 + bh] = np.roll(out[y0:y0 + bh], slab_shift, axis=1)

    return out


def apply_smooth_warp(
    frame: np.ndarray,
    frame_idx: int,
    settings: Settings,
    py_rng: random.Random,
    state: Dict[str, Any]
) -> np.ndarray:
    h, w = frame.shape[:2]
    out = np.empty_like(frame)

    wobble_scale = 1.0
    if state["lock_state"]:
        wobble_scale = max(0.0, 1.0 - settings.lock_strength)

    global_shift_x = 0.0
    global_shift_x += settings.smooth_global_drift
    global_shift_x += math.sin(frame_idx * 0.03 * settings.smooth_global_drift_speed) * abs(settings.smooth_global_drift) * 0.25
    global_shift_x += state["smooth_walk_value"] * settings.smooth_random_walk * 50.0
    global_shift_x += py_rng.uniform(-settings.global_jitter, settings.global_jitter)

    global_shift_y = 0.0
    global_shift_y += settings.smooth_vertical_drift
    global_shift_y += math.sin(frame_idx * 0.03 * settings.smooth_vertical_drift_speed + 1.37) * abs(settings.smooth_vertical_drift) * 0.25

    # apply whole-frame vertical drift first
    vertically_shifted = np.roll(frame, int(global_shift_y), axis=0)

    for y in range(h):
        shift = 0.0

        shift += math.sin(
            y * settings.smooth_freq1 + state["smooth_phase1"]
        ) * settings.smooth_amp1

        shift += math.sin(
            y * settings.smooth_freq2 + state["smooth_phase2"]
        ) * settings.smooth_amp2

        if settings.smooth_vertical_ripple > 0:
            shift += math.sin(
                y * settings.smooth_vertical_ripple_freq
                + frame_idx * 0.03 * settings.smooth_vertical_ripple_speed
            ) * settings.smooth_vertical_ripple

        if settings.smooth_noise > 0:
            shift += math.sin(
                y * 0.007 + state["smooth_noise_phase"]
            ) * settings.smooth_noise * settings.smooth_amp1

        # random vertical spike lobes
        if state["vertical_spike_countdown"] > 0 and state["vertical_spike_centers"]:
            yn = y / max(1, h - 1)
            for center, sign in zip(state["vertical_spike_centers"], state["vertical_spike_signs"]):
                d = yn - center
                spike = math.exp(-(d * d) / max(1e-9, settings.vertical_spike_width))
                shift += sign * spike * settings.vertical_spike_strength

        # optional tiny legacy wobble
        if settings.line_wobble != 0:
            shift += math.sin(
                y * settings.line_frequency + frame_idx * settings.time_frequency * 0.03
            ) * settings.line_wobble * 0.25

        shift += global_shift_x
        shift += py_rng.uniform(-settings.line_random, settings.line_random)
        shift *= wobble_scale

        out[y] = np.roll(vertically_shifted[y], int(shift), axis=0)

    return out


def apply_line_sync_destruction(
    frame: np.ndarray,
    frame_idx: int,
    settings: Settings,
    py_rng: random.Random,
    state: Dict[str, Any]
) -> np.ndarray:
    if settings.warp_mode == "smooth":
        return apply_smooth_warp(frame, frame_idx, settings, py_rng, state)
    return apply_chunked_warp(frame, frame_idx, settings, py_rng, state)


def apply_band_shifts(frame: np.ndarray, settings: Settings, py_rng: random.Random) -> np.ndarray:
    out = frame.copy()
    h, _ = out.shape[:2]
    bands = 0
    while bands < settings.multi_band_max and py_rng.random() < settings.band_shift_chance:
        bh = py_rng.randint(settings.band_min_height, max(settings.band_min_height, settings.band_max_height))
        y0 = py_rng.randint(0, max(0, h - bh))
        shift = py_rng.randint(-settings.band_max_shift, settings.band_max_shift)
        out[y0:y0 + bh] = np.roll(out[y0:y0 + bh], shift, axis=1)
        bands += 1
    return out


def apply_rgb_misalignment(
    frame: np.ndarray,
    settings: Settings,
    py_rng: random.Random,
    frame_idx: int,
    state: Dict[str, Any]
) -> np.ndarray:
    b, g, r = cv2.split(frame)

    if state["rgb_countdown"] <= 0:
        state["rgb_gx"] = int(py_rng.randint(-settings.rgb_shift, settings.rgb_shift) * settings.rgb_split_boost) if settings.rgb_shift > 0 else 0
        state["rgb_rx"] = int(py_rng.randint(-settings.rgb_shift, settings.rgb_shift) * settings.rgb_split_boost) if settings.rgb_shift > 0 else 0
        state["rgb_gy"] = py_rng.randint(-settings.rgb_vertical_shift, settings.rgb_vertical_shift) if settings.rgb_vertical_shift > 0 else 0
        state["rgb_ry"] = py_rng.randint(-settings.rgb_vertical_shift, settings.rgb_vertical_shift) if settings.rgb_vertical_shift > 0 else 0
        state["rgb_countdown"] = py_rng.randint(settings.rgb_persist_min, settings.rgb_persist_max)
    else:
        state["rgb_countdown"] -= 1

    gx = state["rgb_gx"]
    rx = state["rgb_rx"]
    gy = state["rgb_gy"]
    ry = state["rgb_ry"]

    if settings.chroma_phase_oscillation > 0:
        osc = math.sin(frame_idx * 0.03 * settings.chroma_phase_speed) * settings.chroma_phase_oscillation
        gx += int(osc)
        rx -= int(osc * 1.2)

    # extra random per-frame skew so the color split feels less uniform
    gx += py_rng.randint(-max(1, settings.rgb_shift // 3), max(1, settings.rgb_shift // 3))
    rx += py_rng.randint(-max(1, settings.rgb_shift // 3), max(1, settings.rgb_shift // 3))
    gy += py_rng.randint(-max(0, settings.rgb_vertical_shift), max(0, settings.rgb_vertical_shift)) if settings.rgb_vertical_shift > 0 else 0
    ry += py_rng.randint(-max(0, settings.rgb_vertical_shift), max(0, settings.rgb_vertical_shift)) if settings.rgb_vertical_shift > 0 else 0

    g = roll_frame(g, gx, gy)
    r = roll_frame(r, rx, ry)

    if settings.channel_jitter_per_line > 0:
        h = frame.shape[0]
        g2 = np.empty_like(g)
        r2 = np.empty_like(r)
        for y in range(h):
            base = math.sin(y * 0.11 + frame_idx * 0.1)
            s1 = int(base * settings.channel_jitter_per_line + py_rng.uniform(-settings.channel_jitter_per_line, settings.channel_jitter_per_line))
            s2 = int(-base * settings.channel_jitter_per_line + py_rng.uniform(-settings.channel_jitter_per_line, settings.channel_jitter_per_line))
            g2[y] = np.roll(g[y], s1)
            r2[y] = np.roll(r[y], s2)
        g = g2
        r = r2

    return cv2.merge((b, g, r))

def apply_gain_and_brightness(
    frame: np.ndarray,
    settings: Settings,
    py_rng: random.Random,
    frame_idx: int,
    state: Dict[str, Any]
) -> np.ndarray:
    out = frame.astype(np.float32)
    wave = math.sin(frame_idx * 0.03 * settings.brightness_wave_speed) * settings.brightness_wave
    out += wave

    if py_rng.random() < settings.gain_pump_chance:
        out *= py_rng.uniform(settings.gain_min, settings.gain_max)

    if settings.band_brightness_coupling > 0:
        h = out.shape[0]
        for band in state["drift_bands"]:
            y0 = int(band["y"])
            y1 = min(h, y0 + band["height"])
            pulse = math.sin(band["phase"] + frame_idx * 0.05) * settings.band_brightness_coupling
            out[y0:y1] += pulse

    if settings.band_darken > 0:
        h = out.shape[0]
        for band in state["drift_bands"]:
            y0 = int(band["y"])
            y1 = min(h, y0 + band["height"])
            out[y0:y1] *= (1.0 - settings.band_darken)

    return clip_uint8(out)


def apply_tracking_bar(frame: np.ndarray, settings: Settings, py_rng: random.Random) -> np.ndarray:
    out = frame.copy()
    h, _ = out.shape[:2]

    if py_rng.random() < settings.tracking_bar_chance:
        bar_h = min(settings.tracking_bar_height, h)
        y0 = py_rng.randint(0, max(0, h - bar_h))
        shift = py_rng.randint(-settings.tracking_bar_shift, settings.tracking_bar_shift)
        bar = np.roll(out[y0:y0 + bar_h], shift, axis=1)
        bar = clip_uint8(bar.astype(np.float32) * py_rng.uniform(0.3, 0.9))
        out[y0:y0 + bar_h] = bar

    if py_rng.random() < settings.bottom_tear_chance:
        tear_h = max(8, min(h // 8, settings.tracking_bar_height * 2))
        y0 = h - tear_h
        shift = py_rng.randint(-settings.tracking_bar_shift, settings.tracking_bar_shift)
        bottom = np.roll(out[y0:], shift, axis=1)
        bottom = clip_uint8(bottom.astype(np.float32) * py_rng.uniform(0.4, 1.0))
        out[y0:] = bottom

    return out


def apply_dropouts(frame: np.ndarray, settings: Settings, py_rng: random.Random) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    if py_rng.random() < settings.dropout_chance:
        lines = py_rng.randint(settings.dropout_lines_min, max(settings.dropout_lines_min, settings.dropout_lines_max))
        for _ in range(lines):
            y = py_rng.randint(0, h - 1)
            width = min(py_rng.randint(settings.dropout_width_min, max(settings.dropout_width_min, settings.dropout_width_max)), w)
            x = py_rng.randint(0, max(0, w - width))
            kind = py_rng.choice(["black", "white", "noise"])
            if kind == "black":
                out[y:y+1, x:x+width] = 0
            elif kind == "white":
                out[y:y+1, x:x+width] = 255
            else:
                out[y:y+1, x:x+width] = np.random.randint(0, 256, size=(1, width, 3), dtype=np.uint8)

    if py_rng.random() < settings.blank_frame_chance:
        if py_rng.random() < 0.5:
            out[:] = py_rng.randint(0, 20)
        else:
            out = clip_uint8(out.astype(np.float32) * py_rng.uniform(0.02, 0.25))

    return out


def apply_desat_burst(frame: np.ndarray, settings: Settings, py_rng: random.Random) -> np.ndarray:
    if py_rng.random() >= settings.desat_burst_chance:
        return frame
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= py_rng.uniform(0.0, 0.35)
    hsv[..., 2] *= py_rng.uniform(0.7, 1.3)
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def apply_speckle(frame: np.ndarray, settings: Settings, py_rng: random.Random) -> np.ndarray:
    if py_rng.random() >= settings.speckle_chance:
        return frame
    out = frame.copy()
    h, w = out.shape[:2]
    count = py_rng.randint(100, max(100, (h * w) // 120))
    ys = np.random.randint(0, h, size=count)
    xs = np.random.randint(0, w, size=count)
    out[ys, xs] = np.random.randint(0, 256, size=(count, 3), dtype=np.uint8)
    return out


def apply_structured_signal_failures(
    frame: np.ndarray,
    settings: Settings,
    py_rng: random.Random,
    state: Dict[str, Any]
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    should_invert = False

    if settings.invert_luma_default:
        should_invert = state["normal_flash_countdown"] <= 0
    else:
        should_invert = settings.invert_luma_always or state["invert_countdown"] > 0

    if should_invert:
        yuv = cv2.cvtColor(out, cv2.COLOR_BGR2YCrCb)
        yuv[..., 0] = 255 - yuv[..., 0]
        out = cv2.cvtColor(yuv, cv2.COLOR_YCrCb2BGR)

    if py_rng.random() < settings.partial_invert_band_chance:
        bh = py_rng.randint(
            max(6, settings.drift_band_height_min // 2),
            max(8, settings.drift_band_height_max)
        )
        y0 = py_rng.randint(0, max(0, h - bh))
        yuv = cv2.cvtColor(out, cv2.COLOR_BGR2YCrCb)
        yuv[y0:y0 + bh, :, 0] = 255 - yuv[y0:y0 + bh, :, 0]
        out = cv2.cvtColor(yuv, cv2.COLOR_YCrCb2BGR)

    if py_rng.random() < settings.hbi_corruption_chance:
        hh = min(settings.hbi_height, h)
        band = out[:hh].copy()
        band = np.roll(band, py_rng.randint(-w // 3, w // 3), axis=1)
        band = clip_uint8(band.astype(np.float32) * py_rng.uniform(0.3, 1.4))
        out[:hh] = band

    if state["sync_loss_countdown"] > 0:
        top_h = min(max(4, settings.hbi_height * 2), h)
        out[:top_h] = clip_uint8(out[:top_h].astype(np.float32) * 0.15)
        if py_rng.random() < 0.5:
            out[:top_h] = np.roll(out[:top_h], py_rng.randint(-w // 4, w // 4), axis=1)

    if state["mono_countdown"] > 0:
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    return out

def process_video_only(settings: Settings, input_video: str, output_video_no_audio: str) -> None:
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open input video: {input_video}")

    input_fps = cap.get(cv2.CAP_PROP_FPS)
    if not input_fps or input_fps <= 0:
        input_fps = 24.0
    out_fps = settings.fps if settings.fps else input_fps

    ret, first = cap.read()
    if not ret:
        raise RuntimeError("No frames could be read from input.")

    first = resize_if_needed(first, settings)
    h, w = first.shape[:2]
    settings.processing_width = w
    settings.processing_height = h

    fourcc = cv2.VideoWriter_fourcc(*settings.opencv_codec)
    writer = cv2.VideoWriter(output_video_no_audio, fourcc, out_fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open temp output writer: {output_video_no_audio}")

    py_rng = random.Random(settings.seed)
    np_rng = np.random.default_rng(settings.seed)
    state = init_state(settings, h, w, py_rng)

    prev_processed = None
    held_frame = None
    hold_remaining = 0

    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    def do_process(frame: np.ndarray, idx: int) -> np.ndarray:
        update_state(state, settings, idx, h, py_rng)
        f = resize_if_needed(frame, settings)
        f = apply_line_sync_destruction(f, idx, settings, py_rng, state)
        f = apply_band_shifts(f, settings, py_rng)
        f = apply_rgb_misalignment(f, settings, py_rng, idx, state)
        f = apply_rgb_misalignment(f, settings, py_rng, idx, state)
        f = apply_palette_roll(f, settings, py_rng, idx, state)
        f = apply_gain_and_brightness(f, settings, py_rng, idx, state)
        f = apply_gain_and_brightness(f, settings, py_rng, idx, state)
        f = apply_tracking_bar(f, settings, py_rng)
        f = apply_structured_signal_failures(f, settings, py_rng, state)
        f = apply_dropouts(f, settings, py_rng)
        f = apply_desat_burst(f, settings, py_rng)
        f = add_gaussian_noise(f, settings.gaussian_noise, np_rng)
        f = add_salt_pepper(f, settings.saltpepper, np_rng)
        f = apply_speckle(f, settings, py_rng)
        return f

    current = first
    while True:
        if hold_remaining > 0 and held_frame is not None:
            processed = held_frame.copy()
            hold_remaining -= 1
        else:
            processed = do_process(current, frame_idx)

            if prev_processed is not None and settings.ghost > 0:
                processed = clip_uint8(
                    processed.astype(np.float32) * (1.0 - settings.ghost)
                    + prev_processed.astype(np.float32) * settings.ghost
                )

            if prev_processed is not None and py_rng.random() < settings.frame_slip_chance:
                slip_alpha = py_rng.uniform(0.35, 0.85)
                processed = clip_uint8(
                    processed.astype(np.float32) * (1.0 - slip_alpha)
                    + prev_processed.astype(np.float32) * slip_alpha
                )

            if py_rng.random() < settings.freeze_frame_chance:
                held_frame = processed.copy()
                hold_remaining = py_rng.randint(settings.freeze_min, settings.freeze_max)

        writer.write(processed)
        prev_processed = processed.copy()

        if settings.verbose and frame_idx % 25 == 0:
            if total_frames > 0:
                eprint(f"Processed frame {frame_idx}/{total_frames}")
            else:
                eprint(f"Processed frame {frame_idx}")

        ret, nxt = cap.read()
        if not ret:
            break
        current = nxt
        frame_idx += 1

    cap.release()
    writer.release()


def mux_with_audio(settings: Settings, sample_input: str, processed_video_no_audio: str, final_output: str) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-i", processed_video_no_audio,
        "-i", sample_input,
        "-map", "0:v:0",
    ]

    has_audio = sample_has_audio(sample_input)
    settings.sample_has_audio = has_audio

    if has_audio:
        cmd += ["-map", "1:a:0?"]

    cmd += [
        "-c:v", settings.video_codec,
        "-preset", settings.preset_encode,
        "-crf", str(settings.crf),
        "-pix_fmt", "yuv420p",
    ]

    if has_audio:
        if settings.audio_mode == "copy":
            cmd += ["-c:a", "copy"]
        else:
            cmd += ["-c:a", "aac", "-b:a", settings.aac_bitrate]
    else:
        cmd += ["-an"]

    cmd += ["-shortest", final_output]
    run_cmd(cmd)


def main() -> int:
    if not tool_exists("ffmpeg"):
        eprint("Error: ffmpeg not found in PATH.")
        return 2

    args = parse_args()
    
    if not args.output:
        args.output, default_json = build_default_output_paths(args.input)
        if not args.save_settings:
            args.save_settings = default_json
    else:
        if not args.save_settings:
            p = Path(args.output)
            ts = datetime.now().strftime("%Y%m%d-%M%S")
            args.save_settings = str(p.with_name(f"{p.stem}-{ts}.json"))
    settings = args_to_settings(args)

    if settings.load_settings:
        loaded = load_settings_json(settings.load_settings)
        settings = apply_loaded_settings(settings, loaded)

    if settings.seed is None:
        settings.seed = random.SystemRandom().randint(0, 2**31 - 1)

    apply_preset(settings, settings.preset)

    if settings.randomize:
        rng = random.Random(settings.seed)
        randomize_settings(settings, rng)

    if settings.verbose:
        eprint(f"Effective seed: {settings.seed}")

    workdir = tempfile.mkdtemp(prefix="glitch_work_")
    sample_path = None
    temp_video_no_audio = os.path.join(workdir, "processed_video_temp.mp4")

    try:
        effective_input, sample_temp = maybe_extract_sample(settings)
        sample_path = sample_temp
        settings.effective_input = effective_input

        if settings.save_settings:
            save_settings_json(settings.save_settings, settings)

        process_video_only(settings, effective_input, temp_video_no_audio)
        mux_with_audio(settings, effective_input, temp_video_no_audio, settings.output)

        if settings.save_settings:
            save_settings_json(settings.save_settings, settings)

        eprint("Done.")
        eprint(f"Seed used: {settings.seed}")
        if settings.save_settings:
            eprint(f"Settings log: {settings.save_settings}")
        return 0

    except subprocess.CalledProcessError as exc:
        eprint(f"Command failed with exit code {exc.returncode}")
        return exc.returncode or 1
    except Exception as exc:
        eprint(f"Error: {exc}")
        return 1
    finally:
        if settings.keep_temp:
            eprint(f"Keeping temp workdir: {workdir}")
            if sample_path:
                eprint(f"Sample file: {sample_path}")
        else:
            try:
                shutil.rmtree(workdir)
            except OSError:
                pass
            if sample_path:
                try:
                    os.remove(sample_path)
                except OSError:
                    pass


if __name__ == "__main__":
    raise SystemExit(main())