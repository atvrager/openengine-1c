"""Tests and performance benchmarks for Virtual Dynamometer, Inertia Pull, and Exhaust Audio."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Iterator
import wave

import numpy as np
import pytest
from typer.testing import CliRunner

from openengine.cli import app
from openengine.dyno import (
    CrankPressFitSpec,
    DEFAULT_DRUM_INERTIA_KG_M2,
    DEFAULT_SAMPLE_RATE_HZ,
    DynoRoomConditions,
    HelmholtzRunnerSpec,
    InertiaPullConfig,
    InertiaPullResult,
    SteadyDynoConfig,
    SteadyDynoSweepResult,
    build_live_twin_simulator,
    compute_crank_press_fit_margin,
    compute_runner_tuned_speeds,
    compute_sae_j1349_factor,
    run_inertia_dyno_pull,
    run_steady_dyno_sweep,
    synthesize_exhaust_audio,
)
from openengine.reporting.dashboard import generate_interactive_dashboard
from openengine.reporting.summary_md import generate_markdown_summary

KNOCK_SAFE_CEILING: float = 0.88
SAE_STD_MIN: float = 0.96
SAE_STD_MAX: float = 1.01
HOT_HUMID_PRESSURE_PA: float = 95000.0
HOT_HUMID_TEMP_K: float = 313.15
HOT_HUMID_RH: float = 0.85
COLD_DRY_PRESSURE_PA: float = 103000.0
COLD_DRY_TEMP_K: float = 280.15
COLD_DRY_RH: float = 0.15

MIN_PEAK_SAE_HP: float = 11.0
MAX_PEAK_SAE_HP: float = 16.5
MIN_PEAK_SAE_TQ_LBFT: float = 10.0
MAX_PEAK_SAE_TQ_LBFT: float = 14.5
MIN_BSFC_LOWER_G_KWH: float = 240.0
MIN_BSFC_UPPER_G_KWH: float = 360.0
MAX_OVERRUN_BSFC_G_KWH: float = 1500.0
MIN_MECH_EFF: float = 0.65
MAX_MECH_EFF: float = 0.95
HELMHOLTZ_MIN_RPM: float = 4800.0
HELMHOLTZ_MAX_RPM: float = 6400.0

SHORT_BELLMOUTH_M: float = 0.070
SHORT_EXHAUST_M: float = 0.450
MIN_PULL_DURATION_S: float = 8.0
MAX_PULL_DURATION_S: float = 13.0
DOUBLE_FACTOR: float = 2.0
DRUM_SCALE_RPM_START: float = 2000.0
DRUM_SCALE_RPM_MID: float = 4500.0
DRUM_SCALE_RPM_END: float = 7000.0
DRUM_SCALE_OVERRUN_S: float = 0.5
MIN_DRUM_DURATION_RATIO: float = 1.5

MAX_FLYWHEEL_CS: float = 0.045
MIN_BURST_SF: float = 5.0
MIN_PRESS_FIT_SLIP_SF: float = 1.20
MIN_PRESS_FIT_YIELD_SF: float = 1.20
H7_P6_MIN_INTERF_M: float = 1.0e-6
H7_P6_NOM_INTERF_M: float = 18.0e-6
H7_P6_MAX_INTERF_M: float = 35.0e-6
H7_P6_FRICTION_COEFF: float = 0.14
CRANK_TEST_WEB_TEMP_C: float = 95.0

MONO_CHANNELS: int = 1
PCM_16_BYTES: int = 2
PCM_INT16_SCALE: float = 32767.0
MIN_WAV_PEAK_DBFS: float = -1.25
MAX_WAV_PEAK_DBFS: float = -0.50
MIN_OVERRUN_RMS: float = 0.005
WINDOW_START_OFFSET_S: float = 0.25
WINDOW_DURATION_S: float = 1.0
WINDOW_END_TAIL_OFFSET_S: float = 1.25
FFT_MIN_HZ: float = 20.0
FFT_MAX_HZ: float = 900.0
MIN_CENTROID_SHIFT_RATIO: float = 1.20
HUNDRED_FLOAT: float = 100.0
FOUR_THROTTLES: int = 4
TWELVE_RPMS: int = 12
ZERO_FLOAT: float = 0.0

PERF_BASE_RPMS: float = 1400.0
PERF_TOP_RPMS: float = 7000.0
PERF_TWO_POINTS: int = 2
PERF_SAMPLES_SCALE: float = 20000.0
PERF_WAV_DURATION_FACTOR: float = 0.05
MIN_EPSILON: float = 1.0e-12


@pytest.fixture(scope="module")
def steady_result() -> SteadyDynoSweepResult:
  """Cached full steady dyno sweep on live CAD twin for module tests."""
  return run_steady_dyno_sweep()


@pytest.fixture(scope="module")
def pull_result(steady_result: SteadyDynoSweepResult) -> InertiaPullResult:
  """Cached transient inertia pull result coupled to cached steady sweep."""
  return run_inertia_dyno_pull(steady_result)


def test_sae_j1349_atmospheric_correction() -> None:
  """Verify SAE J1349 factor follows air density changes with temperature and pressure."""
  std_cf = compute_sae_j1349_factor(DynoRoomConditions())
  hot_humid_cf = compute_sae_j1349_factor(
      DynoRoomConditions(
          ambient_pressure_pa=HOT_HUMID_PRESSURE_PA,
          ambient_temp_k=HOT_HUMID_TEMP_K,
          relative_humidity=HOT_HUMID_RH,
      )
  )
  cold_dry_cf = compute_sae_j1349_factor(
      DynoRoomConditions(
          ambient_pressure_pa=COLD_DRY_PRESSURE_PA,
          ambient_temp_k=COLD_DRY_TEMP_K,
          relative_humidity=COLD_DRY_RH,
      )
  )
  assert SAE_STD_MIN <= std_cf <= SAE_STD_MAX
  assert hot_humid_cf > std_cf
  assert cold_dry_cf < std_cf


def test_steady_dyno_wot_part_throttle_and_knock_safety(steady_result: SteadyDynoSweepResult) -> None:
  """Verify SAE curves, 87-AKI knock ceiling <= 0.88, Helmholtz VE peak, and BSFC map."""
  assert MIN_PEAK_SAE_HP <= steady_result.peak_sae_power_hp <= MAX_PEAK_SAE_HP
  assert MIN_PEAK_SAE_TQ_LBFT <= steady_result.peak_sae_torque_lb_ft <= MAX_PEAK_SAE_TQ_LBFT
  assert steady_result.peak_torque_rpm < steady_result.peak_power_rpm
  assert HELMHOLTZ_MIN_RPM <= steady_result.helmholtz_tuned_rpm <= HELMHOLTZ_MAX_RPM
  assert HELMHOLTZ_MIN_RPM <= steady_result.exhaust_scavenge_rpm <= HELMHOLTZ_MAX_RPM

  for pt in steady_result.wot_points:
    assert pt.knock_integral <= KNOCK_SAFE_CEILING
    assert MIN_MECH_EFF <= pt.mechanical_efficiency <= MAX_MECH_EFF

  bsfc = steady_result.bsfc_map
  assert len(bsfc.throttles) == FOUR_THROTTLES
  assert len(bsfc.rpms) == TWELVE_RPMS
  assert len(bsfc.bsfc_g_kwh_grid) == FOUR_THROTTLES
  assert len(bsfc.bsfc_g_kwh_grid[0]) == TWELVE_RPMS
  assert MIN_BSFC_LOWER_G_KWH <= bsfc.min_bsfc_g_kwh <= MIN_BSFC_UPPER_G_KWH
  assert max(max(row) for row in bsfc.bsfc_g_kwh_grid) <= MAX_OVERRUN_BSFC_G_KWH
  ve_max_pt = max(steady_result.wot_points, key=lambda p: p.volumetric_efficiency)
  assert ve_max_pt.volumetric_efficiency > steady_result.wot_points[0].volumetric_efficiency


def test_helmholtz_runner_length_shifts_tuned_speed() -> None:
  """Verify shorter intake/exhaust lengths shift Helmholtz & wave tuned RPM upward."""
  base_runner = HelmholtzRunnerSpec.from_cad()
  short_runner = HelmholtzRunnerSpec(
      cad_intake_tract_length_m=base_runner.cad_intake_tract_length_m,
      bellmouth_length_m=SHORT_BELLMOUTH_M,
      intake_runner_diam_m=base_runner.intake_runner_diam_m,
      carb_venturi_diam_m=base_runner.carb_venturi_diam_m,
      exhaust_primary_length_m=SHORT_EXHAUST_M,
      exhaust_pipe_diam_m=base_runner.exhaust_pipe_diam_m,
  )
  sweep_cfg = SteadyDynoConfig(rpms=(HELMHOLTZ_MIN_RPM, HELMHOLTZ_MAX_RPM), throttles=(1.0,))
  geom = build_live_twin_simulator().geom
  in_base, ex_base = compute_runner_tuned_speeds(
      geom=geom,
      runner=base_runner,
      ambient_temp_k=sweep_cfg.room.ambient_temp_k,
  )
  in_short, ex_short = compute_runner_tuned_speeds(
      geom=geom,
      runner=short_runner,
      ambient_temp_k=sweep_cfg.room.ambient_temp_k,
  )
  assert in_short > in_base
  assert ex_short > ex_base


def test_inertia_pull_thermal_viscosity_and_drum_scaling(
    steady_result: SteadyDynoSweepResult,
    pull_result: InertiaPullResult,
) -> None:
  """Verify ~10 s inertia pull, head rise, oil viscosity thinning, flywheel C_s, and drum scaling."""
  assert MIN_PULL_DURATION_S <= pull_result.wot_pull_duration_s <= MAX_PULL_DURATION_S
  assert pull_result.head_temp_end_c > pull_result.head_temp_start_c
  assert pull_result.oil_temp_end_c > pull_result.oil_temp_start_c
  assert pull_result.oil_viscosity_end_cst < pull_result.oil_viscosity_start_cst
  assert pull_result.flywheel_cs_coeff <= MAX_FLYWHEEL_CS
  assert pull_result.flywheel_burst_sf_redline >= MIN_BURST_SF
  assert pull_result.rpm[-1] < pull_result.config.end_rpm

  light_pull = run_inertia_dyno_pull(
      steady_result,
      config=InertiaPullConfig(
          start_rpm=DRUM_SCALE_RPM_START,
          end_rpm=DRUM_SCALE_RPM_MID,
          drum_inertia_kg_m2=DEFAULT_DRUM_INERTIA_KG_M2,
          overrun_duration_s=DRUM_SCALE_OVERRUN_S,
      ),
  )
  heavy_pull = run_inertia_dyno_pull(
      steady_result,
      config=InertiaPullConfig(
          start_rpm=DRUM_SCALE_RPM_START,
          end_rpm=DRUM_SCALE_RPM_MID,
          drum_inertia_kg_m2=DOUBLE_FACTOR * DEFAULT_DRUM_INERTIA_KG_M2,
          overrun_duration_s=DRUM_SCALE_OVERRUN_S,
      ),
  )
  assert heavy_pull.wot_pull_duration_s >= MIN_DRUM_DURATION_RATIO * light_pull.wot_pull_duration_s
  assert DRUM_SCALE_RPM_END > DRUM_SCALE_RPM_MID


def test_crank_press_fit_h6_x6_safe_vs_light_h7_p6_slip(pull_result: InertiaPullResult) -> None:
  """Verify H6/x6 shrink-press fit clears slip >= 1.20 while H7/p6 slips."""
  rep = pull_result.crank_press_fit_report
  assert rep.is_safe is True
  assert rep.min_slip_safety_factor >= MIN_PRESS_FIT_SLIP_SF
  assert rep.bore_yield_safety_factor >= MIN_PRESS_FIT_YIELD_SF

  light_fit = CrankPressFitSpec(
      fit_label="H7/p6",
      min_diametral_interference_m=H7_P6_MIN_INTERF_M,
      nominal_diametral_interference_m=H7_P6_NOM_INTERF_M,
      max_diametral_interference_m=H7_P6_MAX_INTERF_M,
      friction_coeff=H7_P6_FRICTION_COEFF,
  )
  light_rep = compute_crank_press_fit_margin(
      rpm=DRUM_SCALE_RPM_END,
      web_temp_c=CRANK_TEST_WEB_TEMP_C,
      peak_firing_torque_nm=pull_result.peak_firing_torque_nm,
      spec=light_fit,
  )
  assert light_rep.is_safe is False
  assert light_rep.min_slip_safety_factor < 1.0


def _window_spectral_centroid(samples: np.ndarray, fs: int, start_s: float, dur_s: float) -> float:
  """Compute amplitude-weighted spectral centroid within [FFT_MIN_HZ, FFT_MAX_HZ]."""
  i0 = max(0, int(round(start_s * fs)))
  i1 = min(len(samples), i0 + int(round(dur_s * fs)))
  seg = samples[i0:i1] * np.hanning(max(PCM_16_BYTES, i1 - i0))
  freqs = np.fft.rfftfreq(len(seg), d=1.0 / float(fs))
  mags = np.abs(np.fft.rfft(seg))
  band = (freqs >= FFT_MIN_HZ) & (freqs <= FFT_MAX_HZ)
  return float(np.sum(freqs[band] * mags[band]) / max(MIN_EPSILON, float(np.sum(mags[band]))))


def test_exhaust_audio_wav_and_firing_harmonic_shift(
    pull_result: InertiaPullResult,
    steady_result: SteadyDynoSweepResult,
    tmp_path: Path,
) -> None:
  """Verify 44.1 kHz PCM WAV structure, WOT/overrun RMS contrast, and upward frequency shift."""
  wav_out = tmp_path / "exhaust_test.wav"
  res = synthesize_exhaust_audio(pull_result, steady_result, output_wav_path=wav_out)
  assert wav_out.exists()
  assert res.sample_rate_hz == DEFAULT_SAMPLE_RATE_HZ
  assert MIN_WAV_PEAK_DBFS <= res.peak_dbfs <= MAX_WAV_PEAK_DBFS
  assert res.wot_rms > res.overrun_rms > MIN_OVERRUN_RMS
  assert res.end_firing_freq_hz > res.start_firing_freq_hz

  with wave.open(str(wav_out), "rb") as wf:
    assert wf.getnchannels() == MONO_CHANNELS
    assert wf.getsampwidth() == PCM_16_BYTES
    assert wf.getframerate() == DEFAULT_SAMPLE_RATE_HZ
    assert wf.getnframes() == res.num_samples
    raw = wf.readframes(wf.getnframes())

  pcm = np.frombuffer(raw, dtype="<i2").astype(float) / PCM_INT16_SCALE
  assert np.all(np.isfinite(pcm))
  start_cent = _window_spectral_centroid(pcm, res.sample_rate_hz, WINDOW_START_OFFSET_S, WINDOW_DURATION_S)
  end_cent = _window_spectral_centroid(
      pcm,
      res.sample_rate_hz,
      pull_result.wot_pull_duration_s - WINDOW_END_TAIL_OFFSET_S,
      WINDOW_DURATION_S,
  )
  assert end_cent > MIN_CENTROID_SHIFT_RATIO * start_cent


def test_cli_dyno_dashboard_audio_and_summary_integration(tmp_path: Path) -> None:
  """Verify openengine dyno CLI, embedded dashboard audio/BSFC plots, and Markdown summary."""
  runner = CliRunner()
  dyno_dir = tmp_path / "dyno"
  cli_res = runner.invoke(app, ["dyno", "--output-dir", str(dyno_dir)])
  assert cli_res.exit_code == 0, cli_res.output

  json_file = dyno_dir / "dyno_results.json"
  wav_file = dyno_dir / "exhaust_pull_44k1.wav"
  assert json_file.exists() and json_file.stat().st_size > 0
  assert wav_file.exists() and wav_file.stat().st_size > 0

  loaded = json.loads(json_file.read_text(encoding="utf-8"))
  assert loaded["inertia_pull"]["crank_press_fit"]["is_safe"] is True
  assert loaded["steady_sweep"]["bsfc_map"]["min_bsfc_g_kwh"] > ZERO_FLOAT
  assert HUNDRED_FLOAT > loaded["steady_sweep"]["peak_sae_power_hp"] > MIN_PEAK_SAE_HP

  html_out = tmp_path / "reports" / "engineering_dashboard.html"
  generate_interactive_dashboard(
      output_html_path=html_out,
      models_dir=tmp_path / "reports" / "models",
      dyno_json=json_file,
      dyno_wav=wav_file,
  )
  html_txt = html_out.read_text(encoding="utf-8")
  for marker in (
      "Score: 15 / 15 PASS",
      "dyno-main-plot",
      "bsfc-map-plot",
      "dyno-audio-player",
      "data:audio/wav;base64,",
  ):
    assert marker in html_txt

  md_out = tmp_path / "reports" / "summary.md"
  generate_markdown_summary(output_md_path=md_out, dyno_json=json_file)
  md_txt = md_out.read_text(encoding="utf-8")
  assert "## 5. Virtual Dynamometer & Exhaust Audio" in md_txt
  assert "H6/x6" in md_txt


def perf_steady_sweep_scaling(n: int) -> Iterator[None]:
  """Benchmark steady dyno sweep scaling across n RPM points."""
  n_pts = max(PERF_TWO_POINTS, int(n))
  rpms = tuple(float(v) for v in np.linspace(PERF_BASE_RPMS, PERF_TOP_RPMS, n_pts))
  cfg = SteadyDynoConfig(rpms=rpms, throttles=(1.0,))
  yield
  res = run_steady_dyno_sweep(cfg)
  assert len(res.wot_points) == n_pts


def perf_audio_synthesis_scaling(n: int) -> Iterator[None]:
  """Benchmark vectorized audio synthesis across n/scale seconds of clip length."""
  sweep = run_steady_dyno_sweep(SteadyDynoConfig(rpms=(DRUM_SCALE_RPM_START, DRUM_SCALE_RPM_END), throttles=(1.0,)))
  pull = run_inertia_dyno_pull(
      sweep,
      config=InertiaPullConfig(
          start_rpm=DRUM_SCALE_RPM_START,
          end_rpm=DRUM_SCALE_RPM_END,
          overrun_duration_s=max(1.0, float(n) / PERF_SAMPLES_SCALE) * PERF_WAV_DURATION_FACTOR,
      ),
  )
  with tempfile.TemporaryDirectory() as tmp_dir:
    out_wav = Path(tmp_dir) / "bench_dyno.wav"
    yield
    res = synthesize_exhaust_audio(pull, sweep, output_wav_path=out_wav)
    assert res.num_samples > 0
