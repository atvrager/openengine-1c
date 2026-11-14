"""Vectorized 44.1 kHz Exhaust Blowdown, Waveguide Resonance, and Overrun Synthesizer.

Generates a 16-bit mono PCM `.wav` track directly from `InertiaPullResult`
and CAD exhaust primary dimensions (`HelmholtzRunnerSpec`):
  - Periodic 720-deg EVO (135 deg ATDC) blowdown pulses + exhaust expulsion harmonics
  - 1D open-pipe waveguide comb filter (`lfilter`) + Helmholtz muffler biquad
  - Deterministic closed-throttle overrun burble/afterfire crackles
  - Subtle #25 roller chain (18T) and 0.5x SOHC valvetrain mechanical harmonics
  - Peak normalization to -1.0 dBFS with zero per-sample Python loops
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Dict, Optional
import wave
import numpy as np
from scipy.signal import butter, lfilter

from openengine.cad.crankshaft import CRANK_SPROCKET_TEETH
from openengine.dyno.inertia_pull import InertiaPullResult
from openengine.dyno.steady_sweep import (
    DEFAULT_AMBIENT_PRESSURE_PA,
    EXHAUST_GAMMA,
    EXHAUST_PIPE_MEAN_TEMP_K,
    EXHAUST_R_GAS,
    HelmholtzRunnerSpec,
    ROUND_ONE_DECIMAL,
    ROUND_TWO_DECIMALS,
    ROUND_VALUE_DECIMALS,
    SteadyDynoSweepResult,
)

# PCM & DSP constants
DEFAULT_SAMPLE_RATE_HZ: int = 44100
PCM_CHANNELS_MONO: int = 1
PCM_SAMPLE_WIDTH_BYTES: int = 2
PCM_INT16_MAX: int = 32767
TARGET_PEAK_DBFS: float = -1.0
DB_TO_AMP_DIVISOR: float = 20.0
TEN_BASE: float = 10.0

ZERO_FLOAT: float = 0.0
HALF_FACTOR: float = 0.5
TWO_FACTOR: float = 2.0
THREE_FACTOR: float = 3.0
FOUR_FACTOR: float = 4.0
FIVE_FACTOR: float = 5.0
TWO_PI: float = TWO_FACTOR * math.pi
FOUR_PI: float = FOUR_FACTOR * math.pi
SECONDS_PER_MINUTE: float = 60.0
FOUR_STROKE_DIVISOR_HZ: float = 120.0
ONE_STROKE_ORDER: float = 0.5
MIN_EPSILON: float = 1.0e-12

# Thermodynamic & acoustic synthesizer tuning
EVO_ANGLE_RAD: float = math.radians(135.0)
EXHAUST_PHASE_SCALE: float = 1.25
BLOWDOWN_DECAY_RAD: float = 0.48
PULSE_ENV_BASE: float = 0.35
PULSE_ENV_THROTTLE_GAIN: float = 0.65
HARMONIC_2_GAIN: float = 0.42
HARMONIC_3_GAIN: float = 0.18
EXHAUST_JET_NOISE_GAIN: float = 0.16
OVERRUN_THROTTLE_CUTOFF: float = 0.05
OVERRUN_POP_PROBABILITY_HZ: float = 11.0
OVERRUN_POP_DECAY_S: float = 0.018
OVERRUN_POP_GAIN: float = 0.85
OVERRUN_EXPULSION_GAIN: float = 0.22
CHAIN_HARMONIC_GAIN: float = 0.045
VALVETRAIN_HARMONIC_GAIN: float = 0.035

# Filter topology constants
FILTER_ORDER_SECOND: int = 2
WAVEGUIDE_OPEN_END_REFL: float = 0.42
MUFFLER_LOWPASS_HZ: float = 780.0
DC_BLOCK_HIGHPASS_HZ: float = 24.0
FADE_IN_DURATION_S: float = 0.05
FADE_OUT_DURATION_S: float = 0.15
MIN_DELAY_SAMPLES: int = 4
AUDIO_RNG_SEED: int = 20261114
DEFAULT_WAV_PATH: Path = Path("outputs/dyno/exhaust_pull_44k1.wav")


@dataclass(frozen=True, slots=True)
class ExhaustAudioConfig:
  """Audio synthesis sample rate, target normalization ceiling, and seed."""

  sample_rate_hz: int = DEFAULT_SAMPLE_RATE_HZ
  target_peak_dbfs: float = TARGET_PEAK_DBFS
  seed: int = AUDIO_RNG_SEED


@dataclass(frozen=True, slots=True)
class ExhaustAudioResult:
  """Metadata, RMS levels, and spectral markers for synthesized exhaust `.wav`."""

  wav_path: Path
  sample_rate_hz: int
  num_samples: int
  duration_s: float
  peak_dbfs: float
  wot_rms: float
  overrun_rms: float
  start_firing_freq_hz: float
  end_firing_freq_hz: float

  def to_dict(self) -> Dict[str, object]:
    """Serialize exhaust audio metadata into a JSON-safe dictionary."""
    return {
        "wav_path": str(self.wav_path),
        "sample_rate_hz": self.sample_rate_hz,
        "num_samples": self.num_samples,
        "duration_s": round(self.duration_s, ROUND_TWO_DECIMALS),
        "peak_dbfs": round(self.peak_dbfs, ROUND_TWO_DECIMALS),
        "wot_rms": round(self.wot_rms, ROUND_VALUE_DECIMALS),
        "overrun_rms": round(self.overrun_rms, ROUND_VALUE_DECIMALS),
        "start_firing_freq_hz": round(self.start_firing_freq_hz, ROUND_ONE_DECIMAL),
        "end_firing_freq_hz": round(self.end_firing_freq_hz, ROUND_ONE_DECIMAL),
    }


def _synthesize_overrun_pops(
    t_audio: np.ndarray,
    throttle_audio: np.ndarray,
    sample_rate_hz: int,
    rng: np.random.Generator,
) -> np.ndarray:
  """Generate deterministic exponential afterfire burbles during closed-throttle overrun."""
  overrun_mask = throttle_audio <= OVERRUN_THROTTLE_CUTOFF
  n_samples = len(t_audio)
  impulses = np.zeros(n_samples, dtype=float)
  if not np.any(overrun_mask):
    return impulses

  pop_prob = OVERRUN_POP_PROBABILITY_HZ / float(sample_rate_hz)
  hits = (rng.random(n_samples) < pop_prob) & overrun_mask
  impulses[hits] = OVERRUN_POP_GAIN * rng.uniform(HALF_FACTOR, 1.0, size=int(np.sum(hits)))
  alpha = math.exp(-1.0 / max(1.0, OVERRUN_POP_DECAY_S * float(sample_rate_hz)))
  burble_env = lfilter(np.array([1.0]), np.array([1.0, -alpha]), impulses)
  crackles = burble_env * rng.standard_normal(n_samples)
  return crackles


def _apply_exhaust_acoustics(
    raw_source: np.ndarray,
    runner: HelmholtzRunnerSpec,
    sample_rate_hz: int,
) -> np.ndarray:
  """Apply 1D open-pipe waveguide comb, Helmholtz muffler lowpass, and DC highpass."""
  c_ex = math.sqrt(EXHAUST_GAMMA * EXHAUST_R_GAS * EXHAUST_PIPE_MEAN_TEMP_K)
  round_trip_s = (TWO_FACTOR * runner.exhaust_primary_length_m) / max(1.0, c_ex)
  delay_samples = max(MIN_DELAY_SAMPLES, int(round(round_trip_s * float(sample_rate_hz))))

  comb_a = np.zeros(delay_samples + 1, dtype=float)
  comb_a[0] = 1.0
  comb_a[delay_samples] = WAVEGUIDE_OPEN_END_REFL
  comb_b = np.array([1.0], dtype=float)
  waveguide_sig = lfilter(comb_b, comb_a, raw_source)

  b_lp, a_lp = butter(FILTER_ORDER_SECOND, MUFFLER_LOWPASS_HZ, btype="low", fs=float(sample_rate_hz))
  muffled = lfilter(b_lp, a_lp, waveguide_sig)

  b_hp, a_hp = butter(FILTER_ORDER_SECOND, DC_BLOCK_HIGHPASS_HZ, btype="high", fs=float(sample_rate_hz))
  return np.asarray(lfilter(b_hp, a_hp, muffled), dtype=float)


def _apply_edge_fades(signal: np.ndarray, sample_rate_hz: int) -> np.ndarray:
  """Apply smooth linear fade-in and fade-out ramps to prevent boundary clicks."""
  n_samples = len(signal)
  n_in = min(n_samples // int(TWO_FACTOR), int(round(FADE_IN_DURATION_S * float(sample_rate_hz))))
  n_out = min(n_samples // int(TWO_FACTOR), int(round(FADE_OUT_DURATION_S * float(sample_rate_hz))))
  shaped = signal.copy()
  if n_in > 0:
    shaped[:n_in] *= np.linspace(ZERO_FLOAT, 1.0, n_in)
  if n_out > 0:
    shaped[-n_out:] *= np.linspace(1.0, ZERO_FLOAT, n_out)
  return shaped


def _write_pcm16_wav(wav_path: Path, pcm_samples: np.ndarray, sample_rate_hz: int) -> None:
  """Write normalized int16 mono PCM audio frames to disk."""
  wav_path.parent.mkdir(parents=True, exist_ok=True)
  with wave.open(str(wav_path), "wb") as wf:
    wf.setnchannels(PCM_CHANNELS_MONO)
    wf.setsampwidth(PCM_SAMPLE_WIDTH_BYTES)
    wf.setframerate(sample_rate_hz)
    wf.writeframes(pcm_samples.tobytes())


def synthesize_exhaust_audio(
    pull_result: InertiaPullResult,
    steady_result: Optional[SteadyDynoSweepResult] = None,
    output_wav_path: Path = DEFAULT_WAV_PATH,
    runner_spec: Optional[HelmholtzRunnerSpec] = None,
    config: Optional[ExhaustAudioConfig] = None,
) -> ExhaustAudioResult:
  """Synthesize 44.1 kHz mono PCM exhaust `.wav` track from an inertia dyno pull."""
  del steady_result  # Pressures are already carried on pull_result.evo_pressure_pa
  cfg = config or ExhaustAudioConfig()
  runner = runner_spec or HelmholtzRunnerSpec.from_cad()
  rng = np.random.default_rng(cfg.seed)

  fs = cfg.sample_rate_hz
  total_dur_s = max(float(pull_result.time_s[-1]), MIN_EPSILON)
  n_samples = max(MIN_DELAY_SAMPLES, int(round(total_dur_s * float(fs))))
  t_audio = np.arange(n_samples, dtype=float) / float(fs)

  theta_a = np.interp(t_audio, pull_result.time_s, pull_result.theta_rad)
  rpm_a = np.interp(t_audio, pull_result.time_s, pull_result.rpm)
  thr_a = np.interp(t_audio, pull_result.time_s, pull_result.throttle)
  evo_pa_a = np.interp(t_audio, pull_result.time_s, pull_result.evo_pressure_pa)

  cycle_phase = np.mod(theta_a - EVO_ANGLE_RAD, FOUR_PI)
  firing_phase = HALF_FACTOR * cycle_phase
  p_blowdown = np.maximum(ZERO_FLOAT, (evo_pa_a / DEFAULT_AMBIENT_PRESSURE_PA) - 1.0)
  throttle_scale = PULSE_ENV_BASE + (PULSE_ENV_THROTTLE_GAIN * thr_a)

  blowdown = p_blowdown * throttle_scale * np.exp(-cycle_phase / BLOWDOWN_DECAY_RAD)
  expulsion = (thr_a + (OVERRUN_EXPULSION_GAIN * (1.0 - thr_a))) * (
      np.sin(firing_phase)
      + (HARMONIC_2_GAIN * np.sin(TWO_FACTOR * firing_phase))
      + (HARMONIC_3_GAIN * np.sin(THREE_FACTOR * firing_phase))
  )
  jet_noise = EXHAUST_JET_NOISE_GAIN * throttle_scale * (1.0 + blowdown) * rng.standard_normal(n_samples)
  burble = _synthesize_overrun_pops(t_audio, thr_a, fs, rng)

  exhaust_source = (EXHAUST_PHASE_SCALE * blowdown) + expulsion + jet_noise + burble
  tailpipe_sig = _apply_exhaust_acoustics(exhaust_source, runner=runner, sample_rate_hz=fs)

  chain_sig = CHAIN_HARMONIC_GAIN * np.sin(float(CRANK_SPROCKET_TEETH) * theta_a)
  cam_sig = VALVETRAIN_HARMONIC_GAIN * np.sin(ONE_STROKE_ORDER * theta_a)
  combined = _apply_edge_fades(tailpipe_sig + chain_sig + cam_sig, fs)

  max_abs = max(MIN_EPSILON, float(np.max(np.abs(combined))))
  target_linear = TEN_BASE ** (cfg.target_peak_dbfs / DB_TO_AMP_DIVISOR)
  norm_float = combined * (target_linear / max_abs)
  pcm_int16 = np.clip(np.round(norm_float * float(PCM_INT16_MAX)), -PCM_INT16_MAX, PCM_INT16_MAX).astype("<i2")

  wav_path = Path(output_wav_path)
  _write_pcm16_wav(wav_path, pcm_int16, fs)

  norm_actual = pcm_int16.astype(float) / float(PCM_INT16_MAX)
  peak_dbfs = DB_TO_AMP_DIVISOR * math.log10(max(MIN_EPSILON, float(np.max(np.abs(norm_actual)))))
  wot_mask = t_audio <= pull_result.wot_pull_duration_s
  overrun_mask = ~wot_mask
  wot_rms = float(np.sqrt(np.mean(norm_actual[wot_mask] ** TWO_FACTOR))) if np.any(wot_mask) else ZERO_FLOAT
  overrun_rms = float(np.sqrt(np.mean(norm_actual[overrun_mask] ** TWO_FACTOR))) if np.any(overrun_mask) else ZERO_FLOAT

  start_freq = float(rpm_a[0]) / FOUR_STROKE_DIVISOR_HZ
  peak_rpm = float(np.max(rpm_a[wot_mask])) if np.any(wot_mask) else float(rpm_a[0])
  end_freq = peak_rpm / FOUR_STROKE_DIVISOR_HZ

  return ExhaustAudioResult(
      wav_path=wav_path,
      sample_rate_hz=fs,
      num_samples=n_samples,
      duration_s=total_dur_s,
      peak_dbfs=float(peak_dbfs),
      wot_rms=wot_rms,
      overrun_rms=overrun_rms,
      start_firing_freq_hz=start_freq,
      end_firing_freq_hz=end_freq,
  )
