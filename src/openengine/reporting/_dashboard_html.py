"""HTML, CSS, and interactive Three.js / Plotly templates for OpenEngine-1C."""

from html import escape
import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from openengine.reporting._telemetry import (
    DEFAULT_RPM,
    DashboardTelemetry,
    KW_TO_HP,
    MaterialSwatch,
    ScorecardRow,
    ZERO_F,
    _bval,
    _fval,
    _ival,
    _submap,
    _sval,
)

PERCENT_SCALE: float = 100.0
ZERO_INT: int = 0
ONE_INT: int = 1
TWO_INT: int = 2
ROUND_ONE: int = 1
ROUND_TWO: int = 2
MIN_YIELD_PASS_DEFAULT: float = 1.5
THREE_LEN: int = 3

HTML_DOC_TEMPLATE: str = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>OpenEngine-1C Multi-Physics Engineering Dashboard</title>
  <script src="https://cdn.plot.ly/plotly-2.29.1.min.js"></script>
  <script type="importmap">
  {
    "imports": {
      "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
      "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
    }
  }
  </script>
  <style>
__CSS__
  </style>
</head>
<body>
  <header class="top-header">
    <div>
      <h1>OpenEngine-1C Multi-Physics Engineering Digital Twin</h1>
      <p class="subtitle">__SUBTITLE__</p>
    </div>
    <div class="header-badges">
      <span class="badge __SCORECARD_BADGE_CLASS__">Score: __PASSED_COUNT__ / __TOTAL_CHECKS__ PASS</span>
      <span class="badge badge-info">NestWorks C500 + 87-AKI E10</span>
    </div>
  </header>

  <nav class="sticky-nav">
    <div class="nav-links">
      <a href="#sec-twin">1. 3D Twin &amp; P-V</a>
      <a href="#sec-dyno">2. Virtual Dyno</a>
      <a href="#sec-lub">3. 87-AKI &amp; Wet Sump</a>
      <a href="#sec-fea">4. 3D FEA &amp; Scorecard</a>
      <a href="#sec-dfm">5. C500 DFM &amp; 2D Drawings</a>
      <a href="#sec-bom">6. Costed BoM</a>
    </div>
    <div class="artifact-bar">__ARTIFACT_CHIPS__</div>
  </nav>

  <section class="kpi-strip">__KPI_CARDS__</section>

  <!-- Section 1: Dressed 3D WebGL Digital Twin + Synchronized 720 deg P-V Loop -->
  <section id="sec-twin" class="grid-container">
    <div class="card">
      <div class="card-title">
        <span>1:1 Dressed 3D Engine Digital Twin (XCAF Alloy Colors)</span>
        <span id="stroke-badge" class="badge badge-warn">Power / Expansion</span>
      </div>
      <div id="canvas-container-3d" class="viewport-3d">
        <div class="hud-panel">
          <div class="hud-head">
            <span>Kinematic Telemetry</span>
            <span id="mech-status-badge" class="badge badge-pass">SAFE</span>
          </div>
          <div class="hud-grid">
            <div>Piston Z: <strong id="tel-piston-z">0.0 mm</strong></div>
            <div>Int P-V: <strong id="tel-pv-clr">5.95 mm</strong></div>
            <div>Rod Clr: <strong id="tel-rod-clr">16.0 mm</strong></div>
            <div>P<sub>cyl</sub>: <strong id="tel-pcyl">24.1 bar</strong></div>
            <div>In Lift: <strong id="tel-in-lift">0.00 mm</strong></div>
            <div>Ex Lift: <strong id="tel-ex-lift">0.00 mm</strong></div>
            <div>Rod Tilt: <strong id="tel-rod-ang">0.0°</strong></div>
            <div>CR: <strong>__CR_TEXT__</strong></div>
          </div>
        </div>

        <div class="viewport-top-bar">
          <div class="btn-group">
            <button id="btn-view-iso" class="btn btn-sm btn-active">Iso</button>
            <button id="btn-view-front" class="btn btn-sm">Front</button>
            <button id="btn-view-side" class="btn btn-sm">Side</button>
            <button id="btn-view-top" class="btn btn-sm">Top</button>
            <button id="reset-cam-btn" class="btn btn-sm">Reset</button>
          </div>
          <div class="btn-group">
            <button id="xray-btn" class="btn btn-sm btn-active">X-Ray</button>
            <button id="wireframe-btn" class="btn btn-sm">Wire</button>
          </div>
          <div class="btn-group">
            <button id="layer-case-btn" class="btn btn-sm btn-active">Case</button>
            <button id="layer-valvetrain-btn" class="btn btn-sm btn-active">Valvetrain</button>
            <button id="layer-covers-btn" class="btn btn-sm btn-active">Covers</button>
            <button id="layer-cots-btn" class="btn btn-sm btn-active">COTS</button>
            <button id="layer-oil-btn" class="btn btn-sm btn-active">Oil</button>
          </div>
        </div>

        <div class="controls-overlay">
          <button id="play-pause-btn" class="btn btn-sm">Pause</button>
          <select id="speed-select" class="select-sm" title="Animation Speed">
            <option value="0.25">0.2x</option>
            <option value="0.75" selected>0.5x</option>
            <option value="1.5">1.0x</option>
          </select>
          <div class="slider-group">
            <span class="label-strong">CA:</span>
            <input type="range" id="crank-slider" min="0" max="720" value="0" step="1">
            <span id="crank-val" class="ca-readout">0°</span>
          </div>
        </div>
      </div>
      <div class="swatch-legend">__MATERIAL_SWATCHES__</div>
    </div>

    <div class="card">
      <div class="card-title">
        <span>Thermodynamic 720° Indicator Loop (P-V Diagram)</span>
        <span class="badge badge-info">Calibrated IMEP: __IMEP_TEXT__ bar</span>
      </div>
      <div id="pv-plot" class="plot-tall"></div>
    </div>
  </section>

  <!-- Section 2: Virtual Brake Dyno, 2D BSFC Contour Map & Exhaust Audio -->
  <section id="sec-dyno" class="grid-container">
    <div class="card">
      <div class="card-title">
        <span>Virtual Brake Dyno &amp; Inertia Roller Pull (SAE J1349)</span>
        <div class="btn-row">
          <button id="btn-dyno-wot" class="btn btn-sm btn-active">WOT SAE Curves</button>
          <button id="btn-dyno-pull" class="btn btn-sm">Inertia Pull &amp; Thermal</button>
        </div>
      </div>
      <div class="pill-row">
        <span class="badge badge-info">Peak SAE: __PEAK_HP_TEXT__</span>
        <span class="badge badge-info">Peak Torque: __PEAK_TQ_TEXT__</span>
        <span class="badge badge-pass">__DYNO_MECH_BADGE__</span>
      </div>
      <div id="dyno-main-plot" class="plot-med"></div>
      <div class="audio-box">
        <div class="audio-head">
          <strong>Exhaust Acoustic Synthesis (44.1 kHz Waveguide + Overrun)</strong>
          <span id="dyno-audio-readout">Ready | __DYNO_OIL_SHIFT__</span>
        </div>
        <audio id="dyno-audio-player" controls preload="metadata" src="data:audio/wav;base64,__DYNO_WAV_B64__"></audio>
      </div>
    </div>

    <div class="card">
      <div class="card-title">
        <span>2D Brake Specific Fuel Consumption Map (g/kWh)</span>
        <span class="badge badge-info">Min BSFC: __MIN_BSFC_TEXT__</span>
      </div>
      <div id="bsfc-map-plot" class="plot-tall-dyno"></div>
    </div>
  </section>

  <!-- Section 3: US 87-Octane E10 Knock Kinetics & Wet-Sump Splash Lubrication -->
  <section id="sec-lub" class="grid-container">
    <div class="card">
      <div class="card-title">
        <span>4-Zone Mechanical Friction (FMEP) &amp; Thermal Loss Breakdown</span>
        <span class="badge badge-info">Total FMEP: __TOTAL_FMEP_TEXT__</span>
      </div>
      <div id="lub-fmep-plot" class="plot-med"></div>
    </div>

    <div class="card">
      <div class="card-title">
        <span>US 87-AKI E10 Knock Kinetics &amp; Wet-Sump Splash Equilibrium</span>
        <span class="badge __LUB_SAFE_CLASS__">__LUB_SAFE_TEXT__</span>
      </div>
      <table>
        <thead>
          <tr><th>Subsystem Parameter</th><th>Evaluated Value</th><th>Design Limit / Reference</th></tr>
        </thead>
        <tbody>__LUB_ROWS__</tbody>
      </table>
    </div>
  </section>

  <!-- Section 4: 3D CalculiX FEA Visualizer & 15-Check Master Scorecard -->
  <section id="sec-fea" class="grid-container">
    <div class="card">
      <div class="card-title">
        <span>CalculiX C3D10 Quadratic 3D FEA von Mises Stress</span>
        <div class="btn-row">
          <button id="btn-stress-dressed" class="btn btn-sm btn-active">Dressed Engine (114)</button>
          <button id="btn-stress-engine" class="btn btn-sm">Core Engine (9)</button>
          <button id="btn-stress-piston" class="btn btn-sm">Piston (LC-1)</button>
          <button id="btn-stress-conrod" class="btn btn-sm">Con-Rod (LC-3A)</button>
          <select id="fea-model-select" class="select-sm" title="Select FEA Assembly or Component"></select>
        </div>
      </div>
      <div id="fea-canvas-container" class="viewport-3d">
        <div class="hud-panel">
          <div id="fea-meta-title" class="hud-title">Loading FEA Model...</div>
          <div id="fea-meta-stats" class="hud-sub">Extracting boundary stress field...</div>
        </div>
        <div class="viewport-top-bar">
          <div class="btn-group">
            <button id="btn-fea-iso" class="btn btn-sm btn-active">Iso</button>
            <button id="btn-fea-front" class="btn btn-sm">Front</button>
            <button id="btn-fea-side" class="btn btn-sm">Side</button>
            <button id="btn-fea-top" class="btn btn-sm">Top</button>
            <button id="btn-fea-reset" class="btn btn-sm">Reset</button>
            <button id="btn-fea-spin" class="btn btn-sm">Spin</button>
          </div>
          <div class="btn-group">
            <button id="btn-fea-xray" class="btn btn-sm btn-active" title="Toggle translucent housing solids in assemblies">X-Ray</button>
            <button id="btn-fea-mode" class="btn btn-sm" title="Switch between % Material Yield and absolute MPa">Mode: % Yield</button>
          </div>
        </div>
        <div class="colorbar-legend">
          <div class="colorbar-labels"><span id="fea-legend-min">0 MPa</span><span id="fea-legend-max">340 MPa</span></div>
          <div class="colorbar-bar"></div>
          <div id="fea-legend-caption" class="colorbar-caption">von Mises Stress (Turbo)</div>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">
        <span>Master Acceptance Criteria Scorecard (__TOTAL_CHECKS__ Multi-Physics Gates)</span>
        <span class="badge __SCORECARD_BADGE_CLASS__">Score: __PASSED_COUNT__ / __TOTAL_CHECKS__ PASS</span>
      </div>
      <div class="table-scroll">
        <table>
          <thead>
            <tr><th>Subsystem</th><th>Metric</th><th>Target</th><th>Achieved</th><th>Margin</th><th>Status</th></tr>
          </thead>
          <tbody>__SCORECARD_ROWS__</tbody>
        </table>
      </div>
    </div>
  </section>

  <!-- Section 5: NestWorks C500 B-Rep DFM Ledger & 2D Orthographic Setup Sheets -->
  <section id="sec-dfm" class="grid-container">
    <div class="card">
      <div class="card-title">
        <span>NestWorks C500 CNC Live B-Rep DFM &amp; 4th-Axis Fallback Ledger</span>
        <span class="badge __DFM_BADGE_CLASS__">__DFM_BADGE_TEXT__</span>
      </div>
      <div class="table-scroll">
        <table>
          <thead>
            <tr><th>Part ID &amp; Alloy</th><th>Mode</th><th>Envelope (mm)</th><th>R<sub>min</sub></th><th>4th-Axis Fallback Strategy</th><th>Gate</th></tr>
          </thead>
          <tbody>__DFM_ROWS__</tbody>
        </table>
      </div>
    </div>

    <div class="card">
      <div class="card-title">
        <span>2D Orthographic CNC Setup Sheets (SVG Preview &amp; CAD DXF)</span>
        <div class="btn-row">
          <select id="setup-sheet-select" class="select-sm"></select>
          <span id="setup-sheet-links" class="btn-row"></span>
        </div>
      </div>
      <div class="sheet-preview-box">
        <img id="setup-sheet-img" alt="2D Orthographic CNC Setup Sheet Preview" />
      </div>
      <div id="setup-sheet-meta" class="setup-meta-grid"></div>
    </div>
  </section>

  <!-- Section 6: Costed 40-Item McMaster-Carr & Powersports BoM -->
  <section id="sec-bom" class="card section-full">
    <div class="card-title">
      <span>Costed 40-Item McMaster-Carr &amp; Powersports Bill of Materials</span>
      <div class="btn-row">
        <span class="badge badge-pass">Total Build Cost: $__BOM_TOTAL_TEXT__</span>
        <button id="copy-order-pad-btn" class="btn btn-sm btn-active">Copy McMaster Order-Pad</button>
      </div>
    </div>
    <div class="bom-toolbar">
      <div class="pill-row">__BOM_CHANNEL_PILLS__</div>
      <div class="bom-filters">
        <input id="bom-search" type="search" placeholder="Filter part description or SKU..." />
        <select id="bom-channel-filter" class="select-sm">
          <option value="">All Sourcing Channels</option>
          <option value="mcmaster_raw_stock">McMaster Raw Stock</option>
          <option value="mcmaster_fasteners_seals">Fasteners &amp; Seals</option>
          <option value="mcmaster_drive_bearings">Drive &amp; Bearings</option>
          <option value="powersports_cots">Powersports COTS</option>
        </select>
      </div>
    </div>
    <div class="table-scroll-tall">
      <table>
        <thead>
          <tr><th>Item</th><th>Description &amp; Spec</th><th>Vendor &amp; SKU</th><th>Tier</th><th>Req / Packs</th><th>Ext Cost</th><th>CAD Check</th></tr>
        </thead>
        <tbody id="bom-table-body"></tbody>
      </table>
    </div>
  </section>

  <script type="module">
__SCRIPT__
  </script>
</body>
</html>
"""

DASHBOARD_CSS: str = """
    :root {
      --bg-dark: #0b1017;
      --card-bg: #141c28;
      --card-border: #253247;
      --accent-cyan: #00d2ff;
      --accent-green: #00e68a;
      --accent-orange: #ff9d00;
      --accent-red: #ff476f;
      --text-main: #edf3f9;
      --text-muted: #8b9bb4;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    html { scroll-behavior: smooth; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg-dark);
      color: var(--text-main);
      line-height: 1.45;
      padding: 20px;
    }
    .top-header {
      display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 12px;
      margin-bottom: 14px; padding-bottom: 14px; border-bottom: 1px solid var(--card-border);
    }
    h1 { font-size: 22px; font-weight: 700; color: var(--accent-cyan); letter-spacing: -0.3px; }
    .subtitle { color: var(--text-muted); font-size: 13px; margin-top: 2px; }
    .header-badges { display: flex; flex-wrap: wrap; gap: 8px; }
    .sticky-nav {
      position: sticky; top: 8px; z-index: 40; display: flex; flex-wrap: wrap;
      justify-content: space-between; align-items: center; gap: 10px;
      background: rgba(16, 23, 34, 0.92); backdrop-filter: blur(10px);
      border: 1px solid var(--card-border); border-radius: 8px; padding: 8px 12px; margin-bottom: 16px;
    }
    .nav-links, .artifact-bar { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
    .nav-links a {
      color: var(--text-main); text-decoration: none; font-size: 12px; font-weight: 600;
      padding: 4px 9px; border-radius: 5px; background: #1c2738; border: 1px solid transparent;
    }
    .nav-links a:hover { border-color: var(--accent-cyan); color: var(--accent-cyan); }
    .artifact-chip {
      color: var(--accent-cyan); text-decoration: none; font-size: 11px; font-weight: 600;
      padding: 3px 8px; border-radius: 4px; background: rgba(0, 210, 255, 0.12);
      border: 1px solid rgba(0, 210, 255, 0.4);
    }
    .artifact-chip:hover { background: rgba(0, 210, 255, 0.24); }
    .artifact-chip-muted {
      color: var(--text-muted); font-size: 11px; padding: 3px 8px; border-radius: 4px;
      background: rgba(139, 155, 180, 0.08); border: 1px solid var(--card-border);
    }
    .kpi-strip {
      display: grid; grid-template-columns: repeat(auto-fit, minmax(175px, 1fr)); gap: 12px; margin-bottom: 20px;
    }
    .kpi-card {
      background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 8px; padding: 12px 14px;
    }
    .kpi-label { font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.4px; }
    .kpi-value { font-size: 18px; font-weight: 700; color: var(--text-main); margin-top: 2px; }
    .kpi-sub { font-size: 11px; color: var(--accent-cyan); margin-top: 2px; }
    .badge {
      display: inline-block; padding: 3px 9px; border-radius: 4px; font-size: 11px; font-weight: 700; text-transform: uppercase;
    }
    .badge-pass { background: rgba(0, 230, 138, 0.14); color: var(--accent-green); border: 1px solid var(--accent-green); }
    .badge-fail { background: rgba(255, 71, 111, 0.16); color: var(--accent-red); border: 1px solid var(--accent-red); }
    .badge-info { background: rgba(0, 210, 255, 0.14); color: var(--accent-cyan); border: 1px solid var(--accent-cyan); }
    .badge-warn { background: rgba(255, 157, 0, 0.15); color: var(--accent-orange); border: 1px solid var(--accent-orange); }
    .grid-container {
      display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px;
    }
    @media (max-width: 1140px) { .grid-container { grid-template-columns: 1fr; } }
    .card {
      background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 8px; padding: 18px;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
    }
    .section-full { margin-bottom: 24px; }
    .card-title {
      font-size: 15px; font-weight: 600; color: var(--text-main); margin-bottom: 12px;
      display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 8px;
    }
    .viewport-3d {
      width: 100%; height: 480px; background: #080c12; border-radius: 6px; position: relative; overflow: hidden;
    }
    .plot-tall { width: 100%; height: 480px; }
    .plot-tall-dyno { width: 100%; height: 430px; }
    .plot-med { width: 100%; height: 320px; }
    .hud-panel {
      position: absolute; top: 10px; left: 10px; background: rgba(12, 17, 25, 0.88);
      backdrop-filter: blur(8px); padding: 8px 11px; border-radius: 6px; border: 1px solid var(--card-border);
      font-size: 11px; z-index: 10; min-width: 225px;
    }
    .hud-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; color: var(--accent-cyan); font-weight: 700; }
    .hud-title { font-weight: 700; color: var(--accent-cyan); margin-bottom: 2px; }
    .hud-sub { color: var(--text-muted); font-size: 11px; }
    .hud-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 2px 10px; color: var(--text-muted); }
    .hud-grid strong { color: var(--text-main); }
    .viewport-top-bar {
      position: absolute; top: 10px; right: 10px; display: flex; flex-wrap: wrap; justify-content: flex-end;
      gap: 6px; z-index: 10; max-width: calc(100% - 255px);
    }
    .btn-group {
      display: inline-flex; background: rgba(12, 17, 25, 0.88); backdrop-filter: blur(8px);
      border: 1px solid var(--card-border); border-radius: 6px; overflow: hidden;
    }
    .btn-group .btn { border: none; border-radius: 0; border-right: 1px solid var(--card-border); }
    .btn-group .btn:last-child { border-right: none; }
    .btn-row { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
    .pill-row { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 10px; }
    .btn {
      background: #1f2b3d; color: var(--text-main); border: 1px solid var(--card-border);
      padding: 5px 11px; border-radius: 4px; cursor: pointer; font-size: 12px; font-weight: 600; transition: 0.15s;
      text-decoration: none;
    }
    .btn-sm { padding: 4px 9px; font-size: 11px; }
    .btn:hover { background: #2b3b52; border-color: var(--accent-cyan); }
    .btn-active { background: var(--accent-cyan); color: #050910; border-color: var(--accent-cyan); }
    .select-sm {
      background: #1a2434; color: var(--text-main); border: 1px solid var(--card-border);
      border-radius: 4px; padding: 4px 8px; font-size: 11px; font-weight: 600;
    }
    .controls-overlay {
      position: absolute; bottom: 10px; left: 10px; right: 10px; background: rgba(12, 17, 25, 0.88);
      backdrop-filter: blur(8px); padding: 7px 12px; border-radius: 6px; border: 1px solid var(--card-border);
      display: flex; flex-wrap: wrap; align-items: center; gap: 10px; z-index: 10;
    }
    .slider-group { display: flex; align-items: center; gap: 8px; flex-grow: 1; font-size: 12px; color: var(--text-muted); }
    .label-strong { font-weight: 700; color: var(--text-main); }
    .ca-readout { min-width: 40px; font-weight: 700; color: var(--accent-cyan); }
    input[type=range] { flex-grow: 1; accent-color: var(--accent-cyan); }
    .swatch-legend {
      display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; padding-top: 10px; border-top: 1px solid var(--card-border);
    }
    .swatch-chip {
      display: inline-flex; align-items: center; gap: 5px; font-size: 11px; color: var(--text-muted);
      background: #0e1520; border: 1px solid var(--card-border); border-radius: 4px; padding: 2px 7px;
    }
    .swatch-dot { width: 10px; height: 10px; border-radius: 2px; border: 1px solid rgba(255, 255, 255, 0.35); }
    .colorbar-legend {
      position: absolute; bottom: 12px; right: 12px; background: rgba(12, 17, 25, 0.88);
      border: 1px solid var(--card-border); border-radius: 6px; padding: 7px 10px; font-size: 11px; z-index: 10; width: 175px;
    }
    .colorbar-labels { display: flex; justify-content: space-between; font-weight: 600; }
    .colorbar-bar {
      height: 10px; margin: 4px 0; border-radius: 2px;
      background: linear-gradient(to right, #30123b, #4662d8, #35abfb, #1ae4b6, #72fe5e, #c8ef34, #faba39, #f66e19, #cb2807, #7a0403);
    }
    .colorbar-caption { font-size: 9px; color: var(--text-muted); text-align: center; }
    .audio-box {
      margin-top: 12px; padding: 10px 12px; background: #0a0f17; border: 1px solid var(--card-border); border-radius: 6px;
    }
    .audio-head {
      display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; font-size: 12px; color: var(--accent-cyan);
    }
    .audio-head span { color: var(--text-muted); }
    audio { width: 100%; height: 34px; }
    .sheet-preview-box {
      background: #f8fafc; border: 1px solid var(--card-border); border-radius: 6px;
      padding: 8px; min-height: 310px; display: flex; align-items: center; justify-content: center;
    }
    .sheet-preview-box img { max-width: 100%; max-height: 340px; object-fit: contain; }
    .setup-meta-grid {
      display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 10px; font-size: 11px;
    }
    .setup-meta-card {
      background: #0e1520; border: 1px solid var(--card-border); border-radius: 5px; padding: 7px 9px;
    }
    .setup-meta-card span { display: block; font-size: 10px; color: var(--accent-cyan); font-weight: 700; text-transform: uppercase; }
    .bom-toolbar {
      display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 10px; margin-bottom: 10px;
    }
    .bom-filters { display: flex; flex-wrap: wrap; gap: 8px; }
    .bom-filters input {
      background: #0e1520; color: var(--text-main); border: 1px solid var(--card-border);
      border-radius: 4px; padding: 5px 10px; font-size: 12px; min-width: 240px;
    }
    .table-scroll { max-height: 480px; overflow-y: auto; }
    .table-scroll-tall { max-height: 440px; overflow-y: auto; }
    table { width: 100%; border-collapse: collapse; font-size: 12px; text-align: left; }
    th {
      position: sticky; top: 0; background: var(--card-bg); color: var(--text-muted);
      font-weight: 600; padding: 8px 9px; border-bottom: 1px solid var(--card-border);
    }
    td { padding: 7px 9px; border-bottom: 1px solid rgba(37, 50, 71, 0.55); vertical-align: top; }
    tr:last-child td { border-bottom: none; }
    a.sku-link { color: var(--accent-cyan); font-weight: 600; text-decoration: none; }
    a.sku-link:hover { text-decoration: underline; }
"""

DASHBOARD_SCRIPT_TEMPLATE: str = """
    import * as THREE from 'three';
    import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
    import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

    const MODELS_REL_URL = __MODELS_REL_JSON__;
    const KIN_CFG = __KIN_CFG_JSON__;
    const CAD_GLB_DATA = __CAD_GLB_JSON__;
    const trajectoryData = __TRAJ_JSON__;
    const thetaTrace = __THETA_TRACE_JSON__;
    const volTrace = __VOL_TRACE_JSON__;
    const pressTrace = __PRESS_TRACE_JSON__;
    const dynoDataPayload = __DYNO_DATA_JSON__;
    const lubDataPayload = __LUB_DATA_JSON__;
    const pistonStressData = __PISTON_STRESS_JSON__;
    const conrodStressData = __CONROD_STRESS_JSON__;
    const feaAtlasData = __FEA_ATLAS_JSON__;
    const setupSheets = __SETUP_SHEETS_JSON__;
    const bomItems = __BOM_ITEMS_JSON__;
    const orderPadText = __ORDER_PAD_JSON__;
    const availableAssets = new Set(__AVAILABLE_ASSETS_JSON__);

    // --- 1. 720-deg Thermodynamic P-V Loop ---
    const minVol = Math.min(...volTrace);
    const maxVol = Math.max(...volTrace);
    const maxPress = Math.max(...pressTrace);
    Plotly.newPlot('pv-plot', [
      { x: volTrace, y: pressTrace, mode: 'lines', name: 'Indicator Loop', line: { color: '#00d2ff', width: 2.5 } },
      { x: [volTrace[0]], y: [pressTrace[0]], mode: 'markers', name: 'Live CA', marker: { color: '#ff9d00', size: 12 } }
    ], {
      paper_bgcolor: 'transparent',
      plot_bgcolor: '#080c12',
      margin: { l: 55, r: 22, t: 24, b: 44 },
      xaxis: { title: 'Chamber Volume (cm³)', gridcolor: '#1a2332', color: '#8b9bb4', range: [Math.max(0, Math.floor(minVol * 0.75)), Math.ceil(maxVol * 1.1)], fixedrange: true },
      yaxis: { title: 'Cylinder Pressure (bar)', gridcolor: '#1a2332', color: '#8b9bb4', range: [0, Math.max(10, Math.ceil((maxPress * 1.2) / 5) * 5)], fixedrange: true },
      showlegend: false
    }, { responsive: true });

    // --- 2. 1:1 Dressed 3D Engine Digital Twin (XCAF Materials + Full Multi-Body Kinematics) ---
    const kinContainer = document.getElementById('canvas-container-3d');
    const kinScene = new THREE.Scene();
    kinScene.background = new THREE.Color(0x080c12);
    const kinCamera = new THREE.PerspectiveCamera(40, kinContainer.clientWidth / kinContainer.clientHeight, 1, 3000);
    kinCamera.up.set(0, 0, 1);
    kinCamera.position.set(450, -600, 320);

    const kinRenderer = new THREE.WebGLRenderer({ antialias: true });
    kinRenderer.setSize(kinContainer.clientWidth, kinContainer.clientHeight);
    kinRenderer.setPixelRatio(window.devicePixelRatio);
    kinContainer.appendChild(kinRenderer.domElement);

    const kinControls = new OrbitControls(kinCamera, kinRenderer.domElement);
    const camTarget = new THREE.Vector3(0, 0, -55);
    kinControls.target.copy(camTarget);
    kinControls.enableDamping = true;
    kinControls.dampingFactor = 0.08;
    kinControls.update();

    kinScene.add(new THREE.AmbientLight(0xffffff, 1.0));
    const dirLight1 = new THREE.DirectionalLight(0x00d2ff, 1.15);
    dirLight1.position.set(220, -260, 320);
    kinScene.add(dirLight1);
    const dirLight2 = new THREE.DirectionalLight(0xffffff, 0.95);
    dirLight2.position.set(-220, 240, -100);
    kinScene.add(dirLight2);
    const dirLight3 = new THREE.DirectionalLight(0xffb86c, 0.55);
    dirLight3.position.set(0, -320, 120);
    kinScene.add(dirLight3);

    const staticGroup = new THREE.Group();
    const crankPivot = new THREE.Group();
    const conrodPivot = new THREE.Group();
    const pistonRecipGroup = new THREE.Group();
    const valveInGroup = new THREE.Group();
    const valveExGroup = new THREE.Group();
    const camPivot = new THREE.Group();
    const rockerInPivot = new THREE.Group();
    const rockerExPivot = new THREE.Group();
    const crankcaseGroup = new THREE.Group();
    const valvetrainGroup = new THREE.Group();
    const coversGroup = new THREE.Group();
    const cotsGroup = new THREE.Group();
    const oilGroup = new THREE.Group();

    crankPivot.position.set(0, 0, KIN_CFG.crankAxisZMm);
    conrodPivot.position.set(0, 0, KIN_CFG.wristPinZ0Mm);
    camPivot.position.set(0, 0, KIN_CFG.camAxisZMm);
    rockerInPivot.position.set(KIN_CFG.rockerPivotXMm, 0, KIN_CFG.rockerPivotZMm);
    rockerExPivot.position.set(-KIN_CFG.rockerPivotXMm, 0, KIN_CFG.rockerPivotZMm);

    [
      staticGroup, crankPivot, conrodPivot, pistonRecipGroup,
      valveInGroup, valveExGroup, camPivot, rockerInPivot, rockerExPivot,
      crankcaseGroup, valvetrainGroup, coversGroup, cotsGroup, oilGroup
    ].forEach((grp) => kinScene.add(grp));

    const glbLoadedRefs = {};
    const xrayTargetNames = new Set(['cylinder_head', 'cylinder_block', 'cylinder_liner', 'crankcase_group', 'covers_and_aux_group']);
    let isXray = true;
    let isWireframe = false;

    function base64ToArrayBuffer(b64) {
      const bin = window.atob(b64);
      const bytes = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
      return bytes.buffer;
    }

    function applyMeshStyling(model, partName) {
      model.traverse((child) => {
        if (!child.isMesh || !child.material) return;
        const mat = child.material;
        const isFluid = partName === 'oil_pool_group';
        const isTranslucent = mat.transparent || mat.opacity < 0.99;
        mat.side = THREE.DoubleSide;
        mat.metalness = (isFluid || isTranslucent) ? 0.14 : 0.56;
        mat.roughness = isFluid ? 0.18 : (isTranslucent ? 0.24 : 0.34);
        child.userData.defaultOpacity = mat.opacity;
        child.userData.defaultTransparent = isTranslucent;
        child.userData.isXrayTarget = xrayTargetNames.has(partName);
        if (isTranslucent) {
          mat.transparent = true;
          mat.depthWrite = false;
        }
        mat.wireframe = isWireframe;
        mat.needsUpdate = true;
      });
    }

    const gltfLoader = new GLTFLoader();
    function loadCadModel(name, targetGroup, localOffset) {
      const attachScene = (gltf) => {
        const model = gltf.scene;
        model.scale.set(1000.0, 1000.0, 1000.0);
        if (localOffset) model.position.copy(localOffset);
        applyMeshStyling(model, name);
        targetGroup.add(model);
        glbLoadedRefs[name] = model;
      };
      const b64 = CAD_GLB_DATA[name];
      if (b64) {
        gltfLoader.parse(base64ToArrayBuffer(b64), '', attachScene, (err) => console.warn('GLB parse:', name, err));
      } else if (MODELS_REL_URL) {
        gltfLoader.load(`${MODELS_REL_URL}/${name}.glb`, attachScene, undefined, (err) => console.warn('GLB load:', name, err));
      }
    }

    loadCadModel('cylinder_block', staticGroup);
    loadCadModel('cylinder_head', staticGroup);
    loadCadModel('cylinder_liner', staticGroup);
    loadCadModel('crankshaft', crankPivot, new THREE.Vector3(0, 0, -KIN_CFG.crankAxisZMm));
    loadCadModel('crank_covers_spin', crankPivot, new THREE.Vector3(0, 0, -KIN_CFG.crankAxisZMm));
    loadCadModel('crank_vt_spin', crankPivot, new THREE.Vector3(0, 0, -KIN_CFG.crankAxisZMm));
    loadCadModel('connecting_rod', conrodPivot, new THREE.Vector3(0, 0, -KIN_CFG.wristPinZ0Mm));
    loadCadModel('conrod_cots', conrodPivot, new THREE.Vector3(0, 0, -KIN_CFG.wristPinZ0Mm));
    loadCadModel('piston', pistonRecipGroup);
    loadCadModel('piston_cots', pistonRecipGroup);
    loadCadModel('wrist_pin', pistonRecipGroup);
    loadCadModel('valve_intake', valveInGroup);
    loadCadModel('retainer_intake', valveInGroup);
    loadCadModel('valve_exhaust', valveExGroup);
    loadCadModel('retainer_exhaust', valveExGroup);
    loadCadModel('cam_vt_spin', camPivot, new THREE.Vector3(0, 0, -KIN_CFG.camAxisZMm));
    loadCadModel('rocker_intake', rockerInPivot, new THREE.Vector3(-KIN_CFG.rockerPivotXMm, 0, -KIN_CFG.rockerPivotZMm));
    loadCadModel('rocker_exhaust', rockerExPivot, new THREE.Vector3(KIN_CFG.rockerPivotXMm, 0, -KIN_CFG.rockerPivotZMm));
    loadCadModel('crankcase_group', crankcaseGroup);
    loadCadModel('valvetrain_group', valvetrainGroup);
    loadCadModel('covers_and_aux_group', coversGroup);
    loadCadModel('cots_hardware_group', cotsGroup);
    loadCadModel('oil_pool_group', oilGroup);

    const frame0 = trajectoryData.length > 0 ? trajectoryData[0] : {};
    const rockerIn0 = Number(frame0.rocker_intake_deg || 0);
    const rockerEx0 = Number(frame0.rocker_exhaust_deg || 0);
    const valveSin = Math.sin(THREE.MathUtils.degToRad(KIN_CFG.valveTiltDeg));
    const valveCos = Math.cos(THREE.MathUtils.degToRad(KIN_CFG.valveTiltDeg));

    function lerpFrame(a, b, t) {
      if (!a) return b;
      if (!b || t <= 0) return a;
      const mix = (k, def = 0) => Number(a[k] ?? def) + (Number(b[k] ?? def) - Number(a[k] ?? def)) * t;
      return {
        crank_angle_deg: Number(a.crank_angle_deg) + 2.0 * t,
        cam_angle_deg: mix('cam_angle_deg'),
        con_rod_angle_deg: mix('con_rod_angle_deg'),
        wrist_pin_z_mm: mix('wrist_pin_z_mm', KIN_CFG.wristPinZ0Mm),
        piston_crown_z_mm: mix('piston_crown_z_mm'),
        intake_valve_lift_mm: mix('intake_valve_lift_mm'),
        exhaust_valve_lift_mm: mix('exhaust_valve_lift_mm'),
        rocker_intake_deg: mix('rocker_intake_deg'),
        rocker_exhaust_deg: mix('rocker_exhaust_deg'),
        intake_pv_clearance_mm: mix('intake_pv_clearance_mm'),
        rod_to_liner_clearance_mm: mix('rod_to_liner_clearance_mm'),
        is_safe: Boolean(a.is_safe && b.is_safe)
      };
    }

    function updateKinematics(caDeg) {
      const normCA = ((caDeg % 720) + 720) % 720;
      const exactIdx = normCA / 2.0;
      const fIdx = Math.min(Math.floor(exactIdx), Math.max(0, trajectoryData.length - 1));
      const nIdx = trajectoryData.length > 0 ? (fIdx + 1) % trajectoryData.length : 0;
      const frame = trajectoryData.length > 0
        ? lerpFrame(trajectoryData[fIdx], trajectoryData[nIdx], exactIdx - fIdx)
        : { crank_angle_deg: normCA, cam_angle_deg: normCA * 0.5, con_rod_angle_deg: 0, wrist_pin_z_mm: KIN_CFG.wristPinZ0Mm, piston_crown_z_mm: 0, intake_valve_lift_mm: 0, exhaust_valve_lift_mm: 0, rocker_intake_deg: 0, rocker_exhaust_deg: 0, intake_pv_clearance_mm: 6, rod_to_liner_clearance_mm: 20, is_safe: true };

      crankPivot.rotation.y = THREE.MathUtils.degToRad(normCA);
      conrodPivot.position.set(0, 0, frame.wrist_pin_z_mm);
      conrodPivot.rotation.y = -THREE.MathUtils.degToRad(frame.con_rod_angle_deg);
      pistonRecipGroup.position.set(0, 0, frame.wrist_pin_z_mm - KIN_CFG.wristPinZ0Mm);
      camPivot.rotation.y = THREE.MathUtils.degToRad(frame.cam_angle_deg ?? (normCA * KIN_CFG.camToCrankRatio));
      rockerInPivot.rotation.y = THREE.MathUtils.degToRad((frame.rocker_intake_deg || 0) - rockerIn0);
      rockerExPivot.rotation.y = THREE.MathUtils.degToRad((frame.rocker_exhaust_deg || 0) - rockerEx0);

      const dIn = (frame.intake_valve_lift_mm || 0) - KIN_CFG.intakeLift0Mm;
      const dEx = (frame.exhaust_valve_lift_mm || 0) - KIN_CFG.exhaustLift0Mm;
      valveInGroup.position.set(-dIn * valveSin, 0, -dIn * valveCos);
      valveExGroup.position.set(dEx * valveSin, 0, -dEx * valveCos);

      const pIdx = Math.min(Math.floor(normCA / 2.0), volTrace.length - 1);
      Plotly.restyle('pv-plot', { x: [[volTrace[pIdx]]], y: [[pressTrace[pIdx]]] }, [1]);

      document.getElementById('tel-piston-z').innerText = `${frame.piston_crown_z_mm.toFixed(1)} mm`;
      document.getElementById('tel-pv-clr').innerText = `${frame.intake_pv_clearance_mm.toFixed(2)} mm`;
      document.getElementById('tel-rod-clr').innerText = `${frame.rod_to_liner_clearance_mm.toFixed(1)} mm`;
      document.getElementById('tel-pcyl').innerText = `${pressTrace[pIdx].toFixed(1)} bar`;
      document.getElementById('tel-in-lift').innerText = `${(frame.intake_valve_lift_mm || 0).toFixed(2)} mm`;
      document.getElementById('tel-ex-lift').innerText = `${(frame.exhaust_valve_lift_mm || 0).toFixed(2)} mm`;
      document.getElementById('tel-rod-ang').innerText = `${frame.con_rod_angle_deg.toFixed(1)}°`;

      const badge = document.getElementById('stroke-badge');
      const labels = ['Power / Expansion', 'Exhaust Stroke', 'Intake Stroke', 'Compression Stroke'];
      const colors = ['#ff9d00', '#8b9bb4', '#00d2ff', '#a855f7'];
      const sIdx = Math.min(3, Math.floor(normCA / 180));
      badge.innerText = labels[sIdx];
      badge.style.color = colors[sIdx];
      badge.style.borderColor = colors[sIdx];
    }

    let isPlaying = true;
    let currentCA = 0;
    const crankSlider = document.getElementById('crank-slider');
    const crankVal = document.getElementById('crank-val');
    const speedSelect = document.getElementById('speed-select');
    crankSlider.addEventListener('input', (e) => {
      isPlaying = false;
      document.getElementById('play-pause-btn').innerText = 'Play';
      currentCA = parseFloat(e.target.value);
      crankVal.innerText = `${Math.round(currentCA)}°`;
      updateKinematics(currentCA);
    });
    document.getElementById('play-pause-btn').addEventListener('click', () => {
      isPlaying = !isPlaying;
      document.getElementById('play-pause-btn').innerText = isPlaying ? 'Pause' : 'Play';
    });

    document.getElementById('xray-btn').addEventListener('click', () => {
      isXray = !isXray;
      kinScene.traverse((child) => {
        if (!child.isMesh || !child.material || !child.userData.isXrayTarget) return;
        const mat = child.material;
        const opa = isXray ? (child.userData.defaultOpacity ?? 0.25) : 1.0;
        mat.opacity = opa;
        mat.transparent = opa < 0.99;
        mat.depthWrite = !mat.transparent;
        mat.needsUpdate = true;
      });
      document.getElementById('xray-btn').classList.toggle('btn-active', isXray);
    });

    document.getElementById('wireframe-btn').addEventListener('click', () => {
      isWireframe = !isWireframe;
      kinScene.traverse((child) => {
        if (child.isMesh && child.material) { child.material.wireframe = isWireframe; child.material.needsUpdate = true; }
      });
      document.getElementById('wireframe-btn').classList.toggle('btn-active', isWireframe);
    });

    const layerState = { case: true, valvetrain: true, covers: true, cots: true, oil: true };
    function syncLayerVisibility() {
      crankcaseGroup.visible = layerState.case;
      valvetrainGroup.visible = layerState.valvetrain;
      camPivot.visible = layerState.valvetrain;
      rockerInPivot.visible = layerState.valvetrain;
      rockerExPivot.visible = layerState.valvetrain;
      if (glbLoadedRefs.crank_vt_spin) glbLoadedRefs.crank_vt_spin.visible = layerState.valvetrain;
      if (glbLoadedRefs.retainer_intake) glbLoadedRefs.retainer_intake.visible = layerState.valvetrain;
      if (glbLoadedRefs.retainer_exhaust) glbLoadedRefs.retainer_exhaust.visible = layerState.valvetrain;
      coversGroup.visible = layerState.covers;
      if (glbLoadedRefs.crank_covers_spin) glbLoadedRefs.crank_covers_spin.visible = layerState.covers;
      cotsGroup.visible = layerState.cots;
      if (glbLoadedRefs.conrod_cots) glbLoadedRefs.conrod_cots.visible = layerState.cots;
      if (glbLoadedRefs.piston_cots) glbLoadedRefs.piston_cots.visible = layerState.cots;
      oilGroup.visible = layerState.oil;
    }

    [['layer-case-btn', 'case'], ['layer-valvetrain-btn', 'valvetrain'], ['layer-covers-btn', 'covers'], ['layer-cots-btn', 'cots'], ['layer-oil-btn', 'oil']]
      .forEach(([id, key]) => {
        document.getElementById(id).addEventListener('click', () => {
          layerState[key] = !layerState[key];
          document.getElementById(id).classList.toggle('btn-active', layerState[key]);
          syncLayerVisibility();
        });
      });

    function setKinCam(pos, activeId) {
      kinCamera.position.copy(pos);
      kinControls.target.copy(camTarget);
      kinControls.update();
      ['btn-view-iso', 'btn-view-front', 'btn-view-side', 'btn-view-top'].forEach((id) => {
        document.getElementById(id).classList.toggle('btn-active', id === activeId);
      });
    }
    document.getElementById('btn-view-iso').addEventListener('click', () => setKinCam(new THREE.Vector3(450, -600, 320), 'btn-view-iso'));
    document.getElementById('reset-cam-btn').addEventListener('click', () => setKinCam(new THREE.Vector3(450, -600, 320), 'btn-view-iso'));
    document.getElementById('btn-view-front').addEventListener('click', () => setKinCam(new THREE.Vector3(0, -850, -45), 'btn-view-front'));
    document.getElementById('btn-view-side').addEventListener('click', () => setKinCam(new THREE.Vector3(850, 0, -55), 'btn-view-side'));
    document.getElementById('btn-view-top').addEventListener('click', () => setKinCam(new THREE.Vector3(0, -1, 850), 'btn-view-top'));

    // --- 3. Virtual Brake Dyno & BSFC Contour Map ---
    const steadyDyno = dynoDataPayload.steady_sweep || {};
    const pullDyno = dynoDataPayload.inertia_pull || {};
    const pullTrace = pullDyno.trace || {};
    const bsfcDyno = steadyDyno.bsfc_map || {};
    const wotPts = steadyDyno.wot_points || [];
    let dynoViewMode = 'wot';

    function renderDynoWotView() {
      dynoViewMode = 'wot';
      document.getElementById('btn-dyno-wot').classList.add('btn-active');
      document.getElementById('btn-dyno-pull').classList.remove('btn-active');
      Plotly.newPlot('dyno-main-plot', [
        { x: wotPts.map((p) => p.rpm), y: wotPts.map((p) => p.sae_brake_power_hp), mode: 'lines+markers', name: 'SAE Power (hp)', line: { color: '#00d2ff', width: 2.5 } },
        { x: wotPts.map((p) => p.rpm), y: wotPts.map((p) => p.sae_brake_torque_lb_ft), yaxis: 'y2', mode: 'lines+markers', name: 'SAE Torque (lb-ft)', line: { color: '#ff9d00', width: 2.5, dash: 'dash' } }
      ], {
        paper_bgcolor: 'transparent', plot_bgcolor: '#080c12', margin: { l: 55, r: 55, t: 22, b: 42 },
        xaxis: { title: 'Engine Speed (RPM)', gridcolor: '#1a2332', color: '#8b9bb4' },
        yaxis: { title: 'SAE Power (hp)', gridcolor: '#1a2332', color: '#00d2ff' },
        yaxis2: { title: 'SAE Torque (lb-ft)', overlaying: 'y', side: 'right', color: '#ff9d00', showgrid: false },
        legend: { orientation: 'h', x: 0.02, y: 1.09, font: { color: '#c8d6e5', size: 11 } }
      }, { responsive: true });
    }

    function renderDynoPullView(cursorTimeS = null) {
      dynoViewMode = 'pull';
      document.getElementById('btn-dyno-pull').classList.add('btn-active');
      document.getElementById('btn-dyno-wot').classList.remove('btn-active');
      const shapes = cursorTimeS !== null ? [{ type: 'line', x0: cursorTimeS, x1: cursorTimeS, y0: 0, y1: 1, yref: 'paper', line: { color: '#00d2ff', width: 2, dash: 'dot' } }] : [];
      Plotly.newPlot('dyno-main-plot', [
        { x: pullTrace.time_s || [], y: pullTrace.rpm || [], mode: 'lines', name: 'Speed (RPM)', line: { color: '#00e68a', width: 2.5 } },
        { x: pullTrace.time_s || [], y: pullTrace.head_temp_c || [], yaxis: 'y2', mode: 'lines', name: 'Head Temp (C)', line: { color: '#ff476f', width: 2.0 } }
      ], {
        paper_bgcolor: 'transparent', plot_bgcolor: '#080c12', margin: { l: 55, r: 55, t: 22, b: 42 },
        xaxis: { title: 'Elapsed Time (s)', gridcolor: '#1a2332', color: '#8b9bb4' },
        yaxis: { title: 'Engine Speed (RPM)', gridcolor: '#1a2332', color: '#00e68a' },
        yaxis2: { title: 'Head Temp (C)', overlaying: 'y', side: 'right', color: '#ff476f', showgrid: false },
        shapes: shapes,
        legend: { orientation: 'h', x: 0.02, y: 1.09, font: { color: '#c8d6e5', size: 11 } }
      }, { responsive: true });
    }

    renderDynoWotView();
    document.getElementById('btn-dyno-wot').addEventListener('click', () => renderDynoWotView());
    document.getElementById('btn-dyno-pull').addEventListener('click', () => renderDynoPullView());

    const bsfcRpms = bsfcDyno.rpms || [];
    const bsfcThrPct = (bsfcDyno.throttles || []).map((t) => Math.round(t * 1000) / 10);
    const bsfcRawGrid = bsfcDyno.bsfc_g_kwh_grid || [];
    const bmepGrid = bsfcDyno.bmep_bar_grid || [];
    const hpGrid = bsfcDyno.sae_power_hp_grid || [];
    const sparkGrid = bsfcDyno.spark_deg_grid || [];

    let minValidBsfc = Infinity;
    for (let r = 0; r < bsfcRawGrid.length; r++) {
      for (let c = 0; c < (bsfcRawGrid[r] || []).length; c++) {
        const val = Number(bsfcRawGrid[r][c]);
        const bmep = (bmepGrid[r] && bmepGrid[r][c] !== undefined) ? Number(bmepGrid[r][c]) : 1.0;
        if (bmep > 0.15 && Number.isFinite(val) && val > 0 && val < 1000) {
          if (val < minValidBsfc) minValidBsfc = val;
        }
      }
    }
    if (!Number.isFinite(minValidBsfc)) minValidBsfc = 280;
    const bsfcZMin = Math.floor(minValidBsfc / 2) * 2;
    const bsfcZMax = Math.min(420, Math.max(bsfcZMin + 60, Math.ceil((minValidBsfc * 1.45) / 5) * 5));
    const bsfcSpan = Math.max(10, bsfcZMax - bsfcZMin);

    const bsfcZPlot = bsfcRawGrid.map((row, r) => (row || []).map((rawVal, c) => {
      const bmep = (bmepGrid[r] && bmepGrid[r][c] !== undefined) ? Number(bmepGrid[r][c]) : 1.0;
      if (bmep <= 0.15 || rawVal >= bsfcZMax) return bsfcZMax;
      return Number(rawVal);
    }));

    const bsfcCustomData = bsfcRawGrid.map((row, r) => (row || []).map((rawVal, c) => {
      const bmep = (bmepGrid[r] && bmepGrid[r][c] !== undefined) ? Number(bmepGrid[r][c]) : null;
      const hp = (hpGrid[r] && hpGrid[r][c] !== undefined) ? Number(hpGrid[r][c]) : null;
      const spark = (sparkGrid[r] && sparkGrid[r][c] !== undefined) ? Number(sparkGrid[r][c]) : null;
      const isOverrun = (bmep !== null && bmep <= 0.15) || Number(rawVal) >= 1490;
      const bsfcLabel = isOverrun ? 'Overrun / Motoring' : `${Number(rawVal).toFixed(1)} g/kWh`;
      return [
        bsfcLabel,
        bmep !== null ? bmep.toFixed(2) : 'N/A',
        hp !== null ? hp.toFixed(2) : 'N/A',
        spark !== null ? `${spark.toFixed(1)} deg BTDC` : 'N/A'
      ];
    }));

    const bsfcTickVals = [0, 0.2, 0.4, 0.6, 0.8, 1.0].map((f) => Math.round(bsfcZMin + f * bsfcSpan));
    const bsfcTickText = bsfcTickVals.map((v, idx) => idx === bsfcTickVals.length - 1 ? `${v}+` : `${v}`);
    const bsfcColorscale = [
      [0.0, '#00f5a0'],
      [0.14, '#00d2ff'],
      [0.32, '#228be6'],
      [0.50, '#5f3dc4'],
      [0.68, '#f59f00'],
      [0.85, '#f03e3e'],
      [0.999, '#9c1b30'],
      [1.0, '#1e293b']
    ];

    Plotly.newPlot('bsfc-map-plot', [
      {
        type: 'contour',
        x: bsfcRpms,
        y: bsfcThrPct,
        z: bsfcZPlot,
        customdata: bsfcCustomData,
        zmin: bsfcZMin,
        zmax: bsfcZMax,
        colorscale: bsfcColorscale,
        autocontour: false,
        colorbar: {
          title: 'g/kWh',
          tickfont: { color: '#8b9bb4' },
          tickmode: 'array',
          tickvals: bsfcTickVals,
          ticktext: bsfcTickText
        },
        contours: {
          start: bsfcZMin,
          end: bsfcZMax,
          size: 5,
          coloring: 'heatmap',
          showlabels: true,
          labelfont: { color: '#ffffff', size: 10 }
        },
        hovertemplate: '%{x} RPM @ %{y}% Thr<br><b>%{customdata[0]}</b><br>BMEP: %{customdata[1]} bar • Power: %{customdata[2]} hp<br>Spark: %{customdata[3]}<extra></extra>'
      },
      {
        type: 'scatter',
        x: [bsfcDyno.min_bsfc_rpm || 5000],
        y: [(bsfcDyno.min_bsfc_throttle || 1.0) * 100],
        mode: 'markers',
        marker: { color: '#00e68a', size: 12, symbol: 'star', line: { color: '#080c12', width: 1.5 } },
        hovertemplate: 'Min BSFC Island<br>%{x} RPM @ %{y}% Thr<extra></extra>'
      }
    ], {
      paper_bgcolor: 'transparent', plot_bgcolor: '#080c12', margin: { l: 55, r: 22, t: 20, b: 44 },
      xaxis: { title: 'Engine Speed (RPM)', gridcolor: '#1a2332', color: '#8b9bb4' },
      yaxis: { title: 'Throttle Position (%)', gridcolor: '#1a2332', color: '#8b9bb4' },
      showlegend: false
    }, { responsive: true });

    const dynoAudio = document.getElementById('dyno-audio-player');
    const dynoReadout = document.getElementById('dyno-audio-readout');
    const pTimes = pullTrace.time_s || [];
    if (dynoAudio && pTimes.length > 0) {
      dynoAudio.addEventListener('timeupdate', () => {
        const t = dynoAudio.currentTime;
        let idx = 0;
        let best = Math.abs(pTimes[0] - t);
        for (let i = 1; i < pTimes.length; i++) {
          const d = Math.abs(pTimes[i] - t);
          if (d < best) { best = d; idx = i; }
        }
        dynoReadout.innerText = `t=${t.toFixed(2)}s | ${Math.round((pullTrace.rpm || [0])[idx])} RPM | Head ${((pullTrace.head_temp_c || [0])[idx]).toFixed(1)} C`;
        if (dynoViewMode !== 'pull') renderDynoPullView(t);
        else Plotly.relayout('dyno-main-plot', { shapes: [{ type: 'line', x0: t, x1: t, y0: 0, y1: 1, yref: 'paper', line: { color: '#00d2ff', width: 2, dash: 'dot' } }] });
      });
    }

    // --- 4. 4-Zone FMEP & Wet-Sump Friction Chart ---
    const fric = lubDataPayload.friction_zones || {};
    const zoneLabels = ['Piston Rings & Skirt', 'Rolling Crank Bearings', 'SOHC Valvetrain & Chain', 'Splash Dipper & Windage'];
    const zoneFmep = [fric.ring_skirt_fmep_bar || 0, fric.rolling_bearings_fmep_bar || 0, fric.valvetrain_fmep_bar || 0, fric.dipper_windage_fmep_bar || 0];
    const zoneHeat = [fric.ring_skirt_heat_w || 0, fric.rolling_bearings_heat_w || 0, fric.valvetrain_heat_w || 0, fric.dipper_windage_heat_w || 0];
    Plotly.newPlot('lub-fmep-plot', [
      { x: zoneLabels, y: zoneFmep, type: 'bar', name: 'FMEP (bar)', marker: { color: '#00d2ff' } },
      { x: zoneLabels, y: zoneHeat, type: 'bar', yaxis: 'y2', name: 'Friction Power (W)', marker: { color: '#ff9d00' }, opacity: 0.85 }
    ], {
      barmode: 'group', paper_bgcolor: 'transparent', plot_bgcolor: '#080c12', margin: { l: 50, r: 55, t: 25, b: 68 },
      xaxis: { color: '#8b9bb4', gridcolor: '#1a2332' },
      yaxis: { title: 'FMEP (bar)', color: '#00d2ff', gridcolor: '#1a2332' },
      yaxis2: { title: 'Heat Loss (W)', overlaying: 'y', side: 'right', color: '#ff9d00', showgrid: false },
      legend: { orientation: 'h', x: 0.02, y: 1.12, font: { color: '#c8d6e5', size: 11 } }
    }, { responsive: true });

    // --- 5. Interactive 3D FEA Stress Mesh & Full-Engine Atlas Viewer ---
    const feaContainer = document.getElementById('fea-canvas-container');
    const feaScene = new THREE.Scene();
    feaScene.background = new THREE.Color(0x080c12);
    const feaCamera = new THREE.PerspectiveCamera(40, feaContainer.clientWidth / feaContainer.clientHeight, 1, 3000);
    feaCamera.up.set(0, 0, 1);
    const feaRenderer = new THREE.WebGLRenderer({ antialias: true });
    feaRenderer.setSize(feaContainer.clientWidth, feaContainer.clientHeight);
    feaRenderer.setPixelRatio(window.devicePixelRatio);
    feaContainer.appendChild(feaRenderer.domElement);
    const feaControls = new OrbitControls(feaCamera, feaRenderer.domElement);
    feaControls.enableDamping = true;
    feaControls.dampingFactor = 0.08;
    feaScene.add(new THREE.AmbientLight(0xffffff, 0.95));
    const fDir1 = new THREE.DirectionalLight(0xffffff, 0.85); fDir1.position.set(180, -220, 260); feaScene.add(fDir1);
    const fDir2 = new THREE.DirectionalLight(0xffffff, 0.50); fDir2.position.set(-180, 220, -140); feaScene.add(fDir2);

    let currentFeaMesh = null;
    let currentFeaDist = 240.0;
    let feaSpin = false;
    let feaXray = true;
    let feaColorMode = 'yield';
    let activeAtlasModelId = null;

    const feaModelSelect = document.getElementById('fea-model-select');
    const btnStressDressed = document.getElementById('btn-stress-dressed');
    const btnStressEngine = document.getElementById('btn-stress-engine');
    const btnStressPiston = document.getElementById('btn-stress-piston');
    const btnStressConrod = document.getElementById('btn-stress-conrod');
    const btnFeaXray = document.getElementById('btn-fea-xray');
    const btnFeaMode = document.getElementById('btn-fea-mode');

    const TURBO_STOPS = [
      [0.0, [0.188, 0.071, 0.231]],
      [0.11, [0.275, 0.384, 0.847]],
      [0.22, [0.208, 0.671, 0.984]],
      [0.38, [0.102, 0.894, 0.714]],
      [0.54, [0.447, 0.996, 0.369]],
      [0.68, [0.784, 0.937, 0.204]],
      [0.82, [0.980, 0.729, 0.224]],
      [0.92, [0.965, 0.431, 0.098]],
      [0.97, [0.796, 0.157, 0.027]],
      [1.0, [0.478, 0.016, 0.012]]
    ];

    function sampleTurboColor(normVal) {
      const t = Math.max(0, Math.min(1, normVal));
      for (let i = 0; i < TURBO_STOPS.length - 1; i++) {
        const [t0, c0] = TURBO_STOPS[i];
        const [t1, c1] = TURBO_STOPS[i + 1];
        if (t <= t1) {
          const u = (t - t0) / Math.max(1e-6, t1 - t0);
          return [
            c0[0] + u * (c1[0] - c0[0]),
            c0[1] + u * (c1[1] - c0[1]),
            c0[2] + u * (c1[2] - c0[2])
          ];
        }
      }
      return TURBO_STOPS[TURBO_STOPS.length - 1][1];
    }

    function decodeBase64Bytes(b64) {
      const bin = atob(b64);
      const out = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
      return out.buffer;
    }

    const decodedMeshCache = {};
    function getDecodedCanonMesh(meshKey) {
      if (decodedMeshCache[meshKey]) return decodedMeshCache[meshKey];
      const raw = (feaAtlasData.meshes || {})[meshKey];
      if (!raw) return null;
      const pos = new Float32Array(decodeBase64Bytes(raw.pos_b64));
      const vm = new Float32Array(decodeBase64Bytes(raw.vm_b64));
      const idx = new Uint32Array(decodeBase64Bytes(raw.idx_b64));
      const baseGeom = new THREE.BufferGeometry();
      baseGeom.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      baseGeom.setIndex(new THREE.BufferAttribute(idx, 1));
      baseGeom.computeVertexNormals();
      const norm = baseGeom.getAttribute('normal').array;
      const entry = { pos, vm, idx, norm, colorAttrs: {} };
      decodedMeshCache[meshKey] = entry;
      return entry;
    }

    function getVertexColorAttribute(meshKey, decoded, denomMpa, cacheTag) {
      if (decoded.colorAttrs[cacheTag]) return decoded.colorAttrs[cacheTag];
      const colors = new Float32Array(decoded.vm.length * 3);
      const invDenom = 1.0 / Math.max(1e-3, denomMpa);
      for (let i = 0; i < decoded.vm.length; i++) {
        const rgb = sampleTurboColor(decoded.vm[i] * invDenom);
        const o = i * 3;
        colors[o] = rgb[0];
        colors[o + 1] = rgb[1];
        colors[o + 2] = rgb[2];
      }
      const attr = new THREE.BufferAttribute(colors, 3);
      decoded.colorAttrs[cacheTag] = attr;
      return attr;
    }

    function setFeaCam(viewType, activeId) {
      const views = {
        iso: new THREE.Vector3(currentFeaDist * 0.65, -currentFeaDist * 0.72, currentFeaDist * 0.55),
        front: new THREE.Vector3(0, -currentFeaDist, 0),
        side: new THREE.Vector3(currentFeaDist, 0, 0),
        top: new THREE.Vector3(0, 0, currentFeaDist)
      };
      feaCamera.position.copy(views[viewType] || views.iso);
      feaControls.target.set(0, 0, 0);
      feaControls.update();
      ['btn-fea-iso', 'btn-fea-front', 'btn-fea-side', 'btn-fea-top'].forEach((id) => {
        document.getElementById(id).classList.toggle('btn-active', id === activeId);
      });
    }

    function syncFeaToolbarButtons(modelId) {
      btnStressDressed.classList.toggle('btn-active', modelId === 'dressed_engine_assembly');
      btnStressEngine.classList.toggle('btn-active', modelId === 'engine_assembly');
      btnStressPiston.classList.toggle('btn-active', modelId === 'part_piston' || modelId === 'legacy_piston');
      btnStressConrod.classList.toggle('btn-active', modelId === 'part_connecting_rod' || modelId === 'legacy_conrod');
      btnFeaXray.classList.toggle('btn-active', feaXray);
      btnFeaMode.classList.toggle('btn-active', feaColorMode === 'yield');
      btnFeaMode.innerText = feaColorMode === 'yield' ? 'Mode: % Yield' : 'Mode: MPa';
    }

    function renderLegacyFeaMesh(payload, legacyId) {
      if (currentFeaMesh) feaScene.remove(currentFeaMesh);
      let geom;
      if (!payload || !payload.geometry || !payload.geometry.positions || !payload.geometry.positions.length) {
        geom = new THREE.CylinderGeometry(25, 25, 35, 32);
        currentFeaMesh = new THREE.Mesh(geom, new THREE.MeshStandardMaterial({ color: 0x00d2ff, roughness: 0.3 }));
      } else {
        geom = new THREE.BufferGeometry();
        geom.setAttribute('position', new THREE.Float32BufferAttribute(payload.geometry.positions, 3));
        geom.setAttribute('color', new THREE.Float32BufferAttribute(payload.geometry.colors, 3));
        geom.setIndex(payload.geometry.indices);
        geom.computeVertexNormals();
        currentFeaMesh = new THREE.Mesh(geom, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.35, metalness: 0.1, side: THREE.DoubleSide }));
        geom.computeBoundingBox();
        const center = new THREE.Vector3();
        geom.boundingBox.getCenter(center);
        currentFeaMesh.position.sub(center);
      }
      feaScene.add(currentFeaMesh);
      if (payload && payload.metadata) {
        const meta = payload.metadata;
        document.getElementById('fea-meta-title').innerText = meta.title || 'CalculiX FEA von Mises Stress';
        document.getElementById('fea-meta-stats').innerText = `P95: ${Number(meta.p95_stress_mpa || 0).toFixed(1)} MPa • Yield SF: ${Number(meta.yield_safety_factor || 0).toFixed(2)} • Disp: ${Number(meta.max_displacement_mm || 0).toFixed(3)} mm`;
        document.getElementById('fea-legend-min').innerText = '0 MPa';
        document.getElementById('fea-legend-max').innerText = `${Math.round(Number(meta.max_stress_mpa || 340))} MPa`;
        document.getElementById('fea-legend-caption').innerText = 'von Mises Stress (Turbo)';
      }
      geom.computeBoundingSphere();
      const rad = (geom.boundingSphere && geom.boundingSphere.radius > 0) ? geom.boundingSphere.radius : 50.0;
      currentFeaDist = (rad / Math.sin(THREE.MathUtils.degToRad(feaCamera.fov) * 0.5)) * 1.55;
      syncFeaToolbarButtons(legacyId);
      setFeaCam('iso', 'btn-fea-iso');
    }

    const atlasModels = feaAtlasData.models || [];
    const atlasModelMap = {};
    for (const mdl of atlasModels) atlasModelMap[mdl.id] = mdl;

    function renderAtlasModel(modelId, preserveMode = false) {
      const mdl = atlasModelMap[modelId];
      if (!mdl) {
        renderLegacyFeaMesh(modelId === 'legacy_conrod' ? conrodStressData : pistonStressData, modelId);
        return;
      }
      if (!preserveMode) {
        feaColorMode = mdl.part_count > 1 ? 'yield' : 'mpa';
      }
      activeAtlasModelId = modelId;
      if (feaModelSelect.value !== modelId) feaModelSelect.value = modelId;
      if (currentFeaMesh) feaScene.remove(currentFeaMesh);

      const spinRoot = new THREE.Group();
      const modelRoot = new THREE.Group();
      const matOpaque = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.35, metalness: 0.12, side: THREE.DoubleSide });
      const matXray = new THREE.MeshStandardMaterial({
        vertexColors: true, roughness: 0.35, metalness: 0.08, side: THREE.DoubleSide,
        transparent: true, opacity: 0.28, depthWrite: false
      });

      const globalMaxMpa = Math.max(1.0, Number(mdl.max_stress_mpa || 340));
      const partsMetaMap = feaAtlasData.parts || {};

      for (const inst of (mdl.instances || [])) {
        const decoded = getDecodedCanonMesh(inst.mesh_key);
        if (!decoded) continue;
        const pMeta = partsMetaMap[inst.mesh_key] || {};
        const partSy = Math.max(1.0, Number(pMeta.yield_strength_mpa || 310));
        const denomMpa = (feaColorMode === 'yield') ? partSy : globalMaxMpa;
        const cacheTag = (feaColorMode === 'yield') ? `y_${partSy.toFixed(1)}` : `m_${globalMaxMpa.toFixed(1)}`;

        const geom = new THREE.BufferGeometry();
        geom.setAttribute('position', new THREE.BufferAttribute(decoded.pos, 3));
        geom.setAttribute('normal', new THREE.BufferAttribute(decoded.norm, 3));
        geom.setAttribute('color', getVertexColorAttribute(inst.mesh_key, decoded, denomMpa, cacheTag));
        geom.setIndex(new THREE.BufferAttribute(decoded.idx, 1));

        const useXray = (mdl.part_count > 1) && Boolean(inst.is_housing) && feaXray;
        const mesh = new THREE.Mesh(geom, useXray ? matXray : matOpaque);
        if (useXray) mesh.renderOrder = 2;
        const off = inst.offset_mm || [0, 0, 0];
        mesh.position.set(off[0], off[1], off[2]);
        modelRoot.add(mesh);
      }

      const box = new THREE.Box3().setFromObject(modelRoot);
      const center = new THREE.Vector3();
      const size = new THREE.Vector3();
      box.getCenter(center);
      box.getSize(size);
      modelRoot.position.sub(center);
      spinRoot.add(modelRoot);
      currentFeaMesh = spinRoot;
      feaScene.add(currentFeaMesh);

      const sphereRadius = Math.max(25.0, 0.5 * size.length());
      currentFeaDist = (sphereRadius / Math.sin(THREE.MathUtils.degToRad(feaCamera.fov) * 0.5)) * 1.38;

      document.getElementById('fea-meta-title').innerText = mdl.title;
      const govTxt = mdl.limiting_part ? ` • Gov: ${mdl.limiting_part}` : (mdl.material ? ` • ${mdl.material}` : '');
      document.getElementById('fea-meta-stats').innerText =
        `Solids: ${mdl.part_count} • P95: ${Number(mdl.p95_stress_mpa || 0).toFixed(1)} MPa • Min SF: ${Number(mdl.yield_safety_factor || 0).toFixed(2)}${govTxt}`;

      if (feaColorMode === 'yield') {
        document.getElementById('fea-legend-min').innerText = '0% Sy';
        document.getElementById('fea-legend-max').innerText = '100% Sy';
        document.getElementById('fea-legend-caption').innerText = 'Normalized von Mises (% Material Yield)';
      } else {
        document.getElementById('fea-legend-min').innerText = '0 MPa';
        document.getElementById('fea-legend-max').innerText = `${Math.round(globalMaxMpa)} MPa`;
        document.getElementById('fea-legend-caption').innerText = 'von Mises Stress (Turbo)';
      }
      syncFeaToolbarButtons(modelId);
      setFeaCam('iso', 'btn-fea-iso');
    }

    if (atlasModels.length > 0) {
      const catGroups = [
        ['assemblies', 'Multi-Body Engine Assemblies (5)'],
        ['primary', 'Primary Billet & Forged Components'],
        ['cots', 'Valvetrain, Drive & COTS Hardware']
      ];
      for (const [catKey, groupLabel] of catGroups) {
        const items = atlasModels.filter((m) => m.category === catKey);
        if (!items.length) continue;
        const grp = document.createElement('optgroup');
        grp.label = groupLabel;
        for (const item of items) {
          const opt = document.createElement('option');
          opt.value = item.id;
          opt.text = item.part_count > 1 ? `${item.title} (SF ${Number(item.yield_safety_factor).toFixed(2)})` : `${item.title} (SF ${Number(item.yield_safety_factor).toFixed(2)})`;
          grp.appendChild(opt);
        }
        feaModelSelect.appendChild(grp);
      }
      const defaultId = atlasModelMap.dressed_engine_assembly ? 'dressed_engine_assembly' : atlasModels[0].id;
      renderAtlasModel(defaultId, false);
    } else {
      feaModelSelect.style.display = 'none';
      btnStressDressed.style.display = 'none';
      btnStressEngine.style.display = 'none';
      btnFeaXray.style.display = 'none';
      btnFeaMode.style.display = 'none';
      renderLegacyFeaMesh(pistonStressData, 'legacy_piston');
    }

    feaModelSelect.addEventListener('change', (e) => renderAtlasModel(e.target.value, false));
    btnStressDressed.addEventListener('click', () => renderAtlasModel('dressed_engine_assembly', false));
    btnStressEngine.addEventListener('click', () => renderAtlasModel('engine_assembly', false));
    btnStressPiston.addEventListener('click', () => {
      if (atlasModelMap.part_piston) renderAtlasModel('part_piston', false);
      else renderLegacyFeaMesh(pistonStressData, 'legacy_piston');
    });
    btnStressConrod.addEventListener('click', () => {
      if (atlasModelMap.part_connecting_rod) renderAtlasModel('part_connecting_rod', false);
      else renderLegacyFeaMesh(conrodStressData, 'legacy_conrod');
    });
    btnFeaXray.addEventListener('click', () => {
      feaXray = !feaXray;
      if (activeAtlasModelId) renderAtlasModel(activeAtlasModelId, true);
    });
    btnFeaMode.addEventListener('click', () => {
      feaColorMode = feaColorMode === 'yield' ? 'mpa' : 'yield';
      if (activeAtlasModelId) renderAtlasModel(activeAtlasModelId, true);
    });
    document.getElementById('btn-fea-iso').addEventListener('click', () => setFeaCam('iso', 'btn-fea-iso'));
    document.getElementById('btn-fea-reset').addEventListener('click', () => setFeaCam('iso', 'btn-fea-iso'));
    document.getElementById('btn-fea-front').addEventListener('click', () => setFeaCam('front', 'btn-fea-front'));
    document.getElementById('btn-fea-side').addEventListener('click', () => setFeaCam('side', 'btn-fea-side'));
    document.getElementById('btn-fea-top').addEventListener('click', () => setFeaCam('top', 'btn-fea-top'));
    document.getElementById('btn-fea-spin').addEventListener('click', () => {
      feaSpin = !feaSpin;
      document.getElementById('btn-fea-spin').classList.toggle('btn-active', feaSpin);
    });

    // --- 6. 2D Orthographic Setup Sheet Browser ---
    const sheetSelect = document.getElementById('setup-sheet-select');
    const sheetImg = document.getElementById('setup-sheet-img');
    const sheetMeta = document.getElementById('setup-sheet-meta');
    const sheetLinks = document.getElementById('setup-sheet-links');
    const urlRootPrefix = __ASSET_PREFIX_JSON__ === '.' ? 'dfm/drawings' : '../dfm/drawings';

    function showSetupSheet(idx) {
      const sheet = setupSheets[idx];
      if (!sheet) return;
      const svgHref = `${urlRootPrefix}/${sheet.svg_file}`;
      const dxfHref = `${urlRootPrefix}/${sheet.dxf_file}`;
      sheetImg.src = sheet.svg_data_uri || (availableAssets.has(svgHref) ? svgHref : '');
      const dims = Array.isArray(sheet.measured_dims_mm) && sheet.measured_dims_mm.length === 3
        ? `${sheet.measured_dims_mm[0]} x ${sheet.measured_dims_mm[1]} x ${sheet.measured_dims_mm[2]} mm`
        : 'Parametric CAD';
      sheetMeta.innerHTML = [
        `<div class="setup-meta-card"><span>Part &amp; Alloy</span>${sheet.title} (${sheet.material})</div>`,
        `<div class="setup-meta-card"><span>Machining Mode &amp; Stock</span>${sheet.active_mode} | ${dims}</div>`,
        `<div class="setup-meta-card"><span>Work Coordinate Datum</span>${sheet.datum_callout}</div>`,
        `<div class="setup-meta-card"><span>Thread Mill Spec</span>${sheet.thread_callout}</div>`,
        `<div class="setup-meta-card"><span>H7 / Shrink Tolerances</span>${sheet.tolerance_callout}</div>`,
        `<div class="setup-meta-card"><span>Shop Ops &amp; Hand Reaming</span>${sheet.ream_and_ops_notes}</div>`
      ].join('');
      const links = [];
      if (availableAssets.has(svgHref)) links.push(`<a class="artifact-chip" href="${svgHref}" download>SVG</a>`);
      if (availableAssets.has(dxfHref)) links.push(`<a class="artifact-chip" href="${dxfHref}" download>DXF</a>`);
      sheetLinks.innerHTML = links.join('');
    }
    setupSheets.forEach((s, idx) => {
      const opt = document.createElement('option');
      opt.value = String(idx);
      opt.textContent = s.title;
      sheetSelect.appendChild(opt);
    });
    sheetSelect.addEventListener('change', (e) => showSetupSheet(Number(e.target.value)));
    if (setupSheets.length > 0) showSetupSheet(0);

    // --- 7. Costed 40-Item BoM Table Filter + McMaster Order-Pad Copy ---
    const bomTbody = document.getElementById('bom-table-body');
    const bomSearch = document.getElementById('bom-search');
    const bomFilter = document.getElementById('bom-channel-filter');

    function renderBomTable() {
      const q = (bomSearch.value || '').trim().toLowerCase();
      const ch = bomFilter.value;
      const rows = bomItems.filter((item) => {
        if (ch && item.channel !== ch) return false;
        if (!q) return true;
        return `${item.item_id} ${item.description} ${item.sku} ${item.vendor}`.toLowerCase().includes(q);
      });
      bomTbody.innerHTML = rows.map((item) => {
        const tierCls = item.sku_status === 'verified_exact' ? 'badge-pass' : (item.sku_status === 'anchor_table' ? 'badge-info' : 'badge-warn');
        const checkCls = item.is_interface_valid ? 'badge-pass' : 'badge-fail';
        const skuHtml = item.url
          ? `<a class="sku-link" href="${item.url}" target="_blank" rel="noopener noreferrer">${item.sku}</a>`
          : item.sku;
        return `<tr>
          <td><strong>${item.item_id}</strong></td>
          <td>${item.description}<div style="color:#8b9bb4;font-size:11px;">${item.selection_hint || ''}</div></td>
          <td>${item.vendor}<br>${skuHtml}</td>
          <td><span class="badge ${tierCls}">${item.sku_status}</span></td>
          <td>${item.required_qty} / ${item.packs_to_order} pk</td>
          <td><strong>$${Number(item.extended_cost_usd || 0).toFixed(2)}</strong></td>
          <td><span class="badge ${checkCls}">${item.interface_check || 'PASS'}</span></td>
        </tr>`;
      }).join('');
    }
    bomSearch.addEventListener('input', renderBomTable);
    bomFilter.addEventListener('change', renderBomTable);
    renderBomTable();

    const copyBtn = document.getElementById('copy-order-pad-btn');
    copyBtn.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(orderPadText);
        copyBtn.innerText = 'Copied 19 McMaster Lines!';
      } catch (_err) {
        const ta = document.createElement('textarea');
        ta.value = orderPadText;
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
        copyBtn.innerText = 'Copied 19 McMaster Lines!';
      }
      setTimeout(() => { copyBtn.innerText = 'Copy McMaster Order-Pad'; }, 2400);
    });

    function animate() {
      requestAnimationFrame(animate);
      if (isPlaying) {
        currentCA = (currentCA + parseFloat(speedSelect.value || '0.75')) % 720;
        crankSlider.value = currentCA;
        crankVal.innerText = `${Math.round(currentCA)}°`;
        updateKinematics(currentCA);
      }
      kinControls.update();
      kinRenderer.render(kinScene, kinCamera);
      if (feaSpin && currentFeaMesh) currentFeaMesh.rotation.z += 0.008;
      feaControls.update();
      feaRenderer.render(feaScene, feaCamera);
    }
    animate();

    window.addEventListener('resize', () => {
      kinCamera.aspect = kinContainer.clientWidth / kinContainer.clientHeight;
      kinCamera.updateProjectionMatrix();
      kinRenderer.setSize(kinContainer.clientWidth, kinContainer.clientHeight);
      feaCamera.aspect = feaContainer.clientWidth / feaContainer.clientHeight;
      feaCamera.updateProjectionMatrix();
      feaRenderer.setSize(feaContainer.clientWidth, feaContainer.clientHeight);
    });
"""

MODELS_SUBDIR: str = "models"
DRAWINGS_SUBDIR: str = "dfm/drawings"

ARTIFACT_LINK_SPECS: Tuple[Tuple[str, str, bool], ...] = (
    ("Dressed STEP", "cad/dressed_engine_assembly.step", False),
    ("Core STEP", "cad/engine_assembly.step", False),
    ("Exhaust WAV", "dyno/exhaust_pull_44k1.wav", False),
    ("BoM CSV", "bom/bom_costed.csv", False),
    ("Order Pad", "bom/mcmaster_order_pad.txt", False),
    ("C500 DFM JSON", "dfm/c500_dfm_report.json", False),
    ("Summary MD", "summary.md", True),
)


def _artifact_href(asset_url_prefix: str, rel_subpath: str, is_report_root: bool) -> str:
  """Compute relative browser URL for an artifact from either reports/ or _site/."""
  if asset_url_prefix == ".":
    return rel_subpath
  if is_report_root:
    return rel_subpath
  return f"{asset_url_prefix}/{rel_subpath}"


def _artifact_exists_on_disk(out_html_path: Path, asset_url_prefix: str, href: str) -> bool:
  """Check if the target local file referenced by href already exists on disk."""
  if asset_url_prefix == ".":
    resolved = out_html_path.parent / href
  else:
    resolved = out_html_path.parent / href
  try:
    return resolved.resolve().exists() and resolved.resolve().stat().st_size > ZERO_INT
  except OSError:
    return False


def _render_artifact_chips(
    out_html_path: Path,
    asset_url_prefix: str,
    setup_sheets: Sequence[Mapping[str, object]],
) -> Tuple[str, List[str]]:
  """Render download anchor tags for present files and muted labels for missing ones."""
  chips: List[str] = []
  available: List[str] = []
  for label, sub_path, is_root in ARTIFACT_LINK_SPECS:
    href = _artifact_href(asset_url_prefix, sub_path, is_root)
    if _artifact_exists_on_disk(out_html_path, asset_url_prefix, href):
      chips.append(f'<a class="artifact-chip" href="{escape(href)}" download>{escape(label)}</a>')
      available.append(href)
    else:
      chips.append(f'<span class="artifact-chip-muted">{escape(label)}</span>')

  draw_prefix = _artifact_href(asset_url_prefix, DRAWINGS_SUBDIR, False)
  for sheet in setup_sheets:
    svg_href = f"{draw_prefix}/{_sval(sheet, 'svg_file')}"
    dxf_href = f"{draw_prefix}/{_sval(sheet, 'dxf_file')}"
    if _artifact_exists_on_disk(out_html_path, asset_url_prefix, svg_href):
      available.append(svg_href)
    if _artifact_exists_on_disk(out_html_path, asset_url_prefix, dxf_href):
      available.append(dxf_href)
  return "".join(chips), available


def _render_kpi_cards(t: DashboardTelemetry) -> str:
  """Render top summary KPI cards directly from aggregated telemetry."""
  steady = _submap(t.dyno, "steady_sweep")
  lub_cycle = _submap(t.thermo_lub, "cycle")
  lub_therm = _submap(t.thermo_lub, "thermal")
  bom_val = _submap(t.bom, "validation")
  cards = (
      ("Displacement / CR", f"{t.swept_cm3:.1f} cm³", f"Bore {t.bore_mm:.0f} x {t.stroke_mm:.0f} mm | CR {t.compression_ratio:.2f}:1"),
      ("Peak SAE Power", f"{_fval(steady, 'peak_sae_power_hp'):.2f} hp", f"{_fval(steady, 'peak_sae_power_kw'):.2f} kW @ {_fval(steady, 'peak_power_rpm'):.0f} RPM"),
      ("US 87-AKI E10 Knock", f"I = {_fval(lub_cycle, 'knock_integral'):.3f}", f"+{_fval(lub_cycle, 'knock_margin') * PERCENT_SCALE:.1f}% anti-knock @ {_fval(t.thermo_lub, 'rpm', DEFAULT_RPM):.0f} RPM"),
      ("Wet-Sump Oil Temp", f"{_fval(lub_therm, 'equilibrium_oil_temp_c'):.1f} °C", f"{t.oil_vol_l:.3f} L SAE 10W-40 | {t.bdc_dipper_immersion_mm:.1f} mm dip"),
      ("CAD Digital Twin", f"{t.total_part_count} Solids", f"Dry {t.dry_mass_kg:.2f} kg | Wet {t.wet_mass_kg:.2f} kg"),
      ("Costed Build BoM", f"${_fval(bom_val, 'total_cost_usd'):,.2f}", f"{_ival(bom_val, 'total_items')} items ({_ival(bom_val, 'verified_exact_mcmaster_count')} Exact SKUs)"),
  )
  return "".join(
      f'<div class="kpi-card"><div class="kpi-label">{escape(lbl)}</div>'
      f'<div class="kpi-value">{escape(val)}</div>'
      f'<div class="kpi-sub">{escape(sub)}</div></div>'
      for lbl, val, sub in cards
  )


def _render_material_swatches(swatches: Sequence[MaterialSwatch]) -> str:
  """Render material swatch chips showing XCAF alloy palette and opacity."""
  items: List[str] = []
  for sw in swatches:
    opa_note = f" ({int(round(sw.opacity * PERCENT_SCALE))}%)" if sw.opacity < ONE_INT else ""
    items.append(
        f'<span class="swatch-chip"><span class="swatch-dot" style="background:{sw.hex_color};opacity:{sw.opacity:.2f};"></span>'
        f'{escape(sw.label)}{opa_note}</span>'
    )
  return "".join(items)


def _render_scorecard_rows(scorecard: Sequence[ScorecardRow]) -> str:
  """Render Master Acceptance Scorecard rows."""
  rows: List[str] = []
  for row in scorecard:
    cls = "badge-pass" if row.passed else "badge-fail"
    txt = "PASS" if row.passed else "FAIL"
    rows.append(
        f"<tr><td><strong>{escape(row.subsystem)}</strong></td><td>{escape(row.metric)}</td>"
        f"<td>{escape(row.target)}</td><td><strong>{escape(row.achieved)}</strong></td>"
        f"<td>{escape(row.margin)}</td><td><span class=\"badge {cls}\">{txt}</span></td></tr>"
    )
  return "".join(rows)


def _render_lub_rows(t: DashboardTelemetry) -> str:
  """Render US 87-AKI knock kinetics and wet-sump splash lubrication table rows."""
  fuel = _submap(t.thermo_lub, "fuel")
  cycle = _submap(t.thermo_lub, "cycle")
  splash = _submap(t.thermo_lub, "splash")
  therm = _submap(t.thermo_lub, "thermal")
  rows = (
      ("Fuel Blend (US Pump E10)", f"{_fval(fuel, 'aki'):.0f} AKI ({_fval(fuel, 'ron'):.1f} RON / {_fval(fuel, 'mon'):.1f} MON)", f"{_fval(fuel, 'ethanol_pct'):.0f}% Ethanol, AFR {_fval(fuel, 'stoich_afr'):.1f}"),
      ("Livengood-Wu Knock Integral", f"{_fval(cycle, 'knock_integral'):.4f} (Margin +{_fval(cycle, 'knock_margin') * PERCENT_SCALE:.1f}%)", "Must remain < 1.000 at WOT"),
      ("Cycle Brake Output @ Rated", f"{_fval(cycle, 'brake_power_kw'):.2f} kW ({_fval(cycle, 'brake_power_kw') * KW_TO_HP:.2f} hp)", f"BSFC {_fval(cycle, 'bsfc_g_kwh'):.1f} g/kWh | Mech Eff {_fval(cycle, 'mechanical_efficiency') * PERCENT_SCALE:.1f}%"),
      ("Sump Dipper Immersion (Static -> Dyn)", f"{_fval(splash, 'static_immersion_mm'):.2f} -> {_fval(splash, 'dynamic_immersion_mm'):.2f} mm", f"{_fval(splash, 'oil_in_flight_ml'):.1f} mL oil suspended in flight"),
      ("Ballistic Throw Velocity / Apex", f"{_fval(splash, 'tip_throw_velocity_m_s'):.2f} m/s / {_fval(splash, 'droplet_apex_height_m'):.2f} m", f"Reaches Under-Crown: {_bval(splash, 'reaches_piston_undercrown')}"),
      ("#25 Chain Carry vs Gravity Drain", f"{_fval(splash, 'chain_oil_carry_g_s'):.2f} g/s carry", f"{_fval(splash, 'tunnel_drain_capacity_g_s'):.1f} g/s cap ({_fval(splash, 'drain_safety_ratio'):.1f}x safety)"),
      ("Equilibrium Sump Oil Temperature", f"{_fval(therm, 'equilibrium_oil_temp_c'):.1f} °C", f"Ceiling {_fval(therm, 'max_allowable_oil_temp_c'):.0f} °C (Margin +{_fval(therm, 'oil_temp_margin_c'):.1f} °C)"),
      ("Splash Piston Cooling & Barrel Relief", f"Crown {_fval(therm, 'cooled_piston_crown_temp_c'):.1f} / Skirt {_fval(therm, 'cooled_piston_skirt_temp_c'):.1f} °C", f"HTC {_fval(therm, 'undercrown_htc_w_m2_k'):.0f} W/(m²·K) | Barrel {_fval(therm, 'adjusted_barrel_heat_budget_w'):.0f} W"),
  )
  return "".join(f"<tr><td><strong>{escape(k)}</strong></td><td>{escape(v)}</td><td>{escape(r)}</td></tr>" for k, v, r in rows)


def _format_dfm_audit_row(item: Mapping[str, object]) -> str:
  """Format one NestWorks C500 B-Rep part audit record into an HTML table row."""
  dims = item.get("measured_dims_mm")
  lim = item.get("envelope_limit_mm")
  dim_txt = "Verified"
  if isinstance(dims, list) and isinstance(lim, list) and len(dims) >= THREE_LEN and len(lim) >= THREE_LEN:
    dim_txt = (
        f"{float(dims[0]):.0f}x{float(dims[1]):.0f}x{float(dims[TWO_INT]):.0f} / "
        f"{float(lim[0]):.0f}x{float(lim[1]):.0f}x{float(lim[TWO_INT]):.0f}"
    )
  ok = _bval(item, "is_valid")
  cls = "badge-pass" if ok else "badge-fail"
  gate = "FALLBACK" if _bval(item, "uses_fallback") else ("PASS" if ok else "FAIL")
  return (
      f"<tr><td><strong>{escape(_sval(item, 'part_id'))}</strong>"
      f"<div style='color:#8b9bb4;font-size:10px;'>{escape(_sval(item, 'material'))}</div></td>"
      f"<td>{escape(_sval(item, 'active_mode'))}</td><td>{dim_txt}</td>"
      f"<td>{_fval(item, 'min_cylindrical_radius_mm'):.1f} mm</td>"
      f"<td>{escape(_sval(item, 'fallback_strategy'))}</td>"
      f"<td><span class='badge {cls}'>{gate}</span></td></tr>"
  )


def _render_dfm_rows(dfm: Mapping[str, object]) -> str:
  """Render NestWorks C500 10-part live B-Rep audit table rows."""
  raw_audits = dfm.get("part_audits")
  audits = raw_audits if isinstance(raw_audits, list) else []
  return "".join(_format_dfm_audit_row(item) for item in audits if isinstance(item, dict))


def _render_bom_channel_pills(bom: Mapping[str, object]) -> str:
  """Render per-channel BoM cost rollup pills."""
  val = _submap(bom, "validation")
  costs = _submap(val, "channel_costs_usd")
  labels = (
      ("mcmaster_raw_stock", "Raw Stock"),
      ("mcmaster_fasteners_seals", "Fasteners & Seals"),
      ("mcmaster_drive_bearings", "Drive & Bearings"),
      ("powersports_cots", "Powersports COTS"),
  )
  pills = [f'<span class="badge badge-info">{lbl}: ${_fval(costs, key):,.2f}</span>' for key, lbl in labels]
  pills.append(f'<span class="badge badge-pass">{_ival(val, "verified_exact_mcmaster_count")} Exact / {_ival(val, "anchor_table_mcmaster_count")} Anchor / {_ival(val, "powersports_cots_count")} COTS</span>')
  return "".join(pills)


def render_dashboard_html(
    t: DashboardTelemetry,
    out_html_path: Path,
    asset_url_prefix: str = "..",
) -> str:
  """Populate the complete interactive dashboard HTML document from DashboardTelemetry."""
  calib = _submap(t.closed_loop, "calibrated")
  lub_cycle = _submap(t.thermo_lub, "cycle")
  lub_fric = _submap(t.thermo_lub, "friction_zones")
  lub_therm = _submap(t.thermo_lub, "thermal")
  steady = _submap(t.dyno, "steady_sweep")
  pull = _submap(t.dyno, "inertia_pull")
  bsfc = _submap(steady, "bsfc_map")
  bom_val = _submap(t.bom, "validation")

  imep_bar = _fval(calib, "imep_bar", _fval(lub_cycle, "imep_bar"))
  peak_hp_txt = f"{_fval(steady, 'peak_sae_power_hp'):.2f} hp @ {_fval(steady, 'peak_power_rpm'):.0f} RPM"
  peak_tq_txt = f"{_fval(steady, 'peak_sae_torque_lb_ft'):.2f} lb-ft @ {_fval(steady, 'peak_torque_rpm'):.0f} RPM"
  mech_badge = f"Pull {_fval(pull, 'wot_pull_duration_s'):.2f}s | H6/x6 SF {_fval(pull, 'min_crank_press_fit_sf'):.2f} | Burst {_fval(pull, 'flywheel_burst_sf_redline'):.1f}"
  oil_shift = f"Oil {_fval(pull, 'oil_viscosity_start_cst'):.1f} -> {_fval(pull, 'oil_viscosity_end_cst'):.1f} cSt"
  min_bsfc_txt = f"{_fval(bsfc, 'min_bsfc_g_kwh'):.1f} g/kWh @ {_fval(bsfc, 'min_bsfc_rpm'):.0f} RPM ({_fval(bsfc, 'min_bsfc_throttle') * PERCENT_SCALE:.0f}% thr)"
  lub_safe = _bval(lub_therm, "is_thermal_safe") and not _bval(lub_cycle, "is_knocking")
  dfm_passed = _bval(t.dfm, "all_passed")
  dfm_mode_txt = "10 / 10 PASS (4th-Axis Active)" if _bval(t.dfm, "has_fourth_axis") else "10 / 10 PASS (3-Axis Fallback)"

  artifact_chips, available_assets = _render_artifact_chips(out_html_path, asset_url_prefix, t.setup_sheets)

  script_replacements: Mapping[str, str] = {
      "__MODELS_REL_JSON__": json.dumps(MODELS_SUBDIR),
      "__KIN_CFG_JSON__": json.dumps(t.kinematic_config),
      "__CAD_GLB_JSON__": json.dumps(t.cad_glb_b64),
      "__TRAJ_JSON__": t.trajectory_json,
      "__THETA_TRACE_JSON__": json.dumps(t.pv_theta_deg),
      "__VOL_TRACE_JSON__": json.dumps(t.pv_volume_cm3),
      "__PRESS_TRACE_JSON__": json.dumps(t.pv_pressure_bar),
      "__DYNO_DATA_JSON__": json.dumps(t.dyno),
      "__LUB_DATA_JSON__": json.dumps(t.thermo_lub),
      "__PISTON_STRESS_JSON__": t.piston_stress_json,
      "__CONROD_STRESS_JSON__": t.conrod_stress_json,
      "__FEA_ATLAS_JSON__": t.fea_atlas_webgl_json,
      "__SETUP_SHEETS_JSON__": json.dumps(t.setup_sheets),
      "__BOM_ITEMS_JSON__": json.dumps(t.bom.get("items", [])),
      "__ORDER_PAD_JSON__": json.dumps(t.order_pad_text),
      "__AVAILABLE_ASSETS_JSON__": json.dumps(available_assets),
      "__ASSET_PREFIX_JSON__": json.dumps(asset_url_prefix),
  }
  script_body = DASHBOARD_SCRIPT_TEMPLATE
  for token, value in script_replacements.items():
    script_body = script_body.replace(token, value)

  html_replacements: Mapping[str, str] = {
      "__CSS__": DASHBOARD_CSS,
      "__SUBTITLE__": f"Single-Cylinder 4-Stroke SOHC • {t.bore_mm:.1f} mm Bore x {t.stroke_mm:.1f} mm Stroke ({t.swept_cm3:.1f} cm³) • CAD CR {t.compression_ratio:.2f}:1 • {t.total_part_count} Solids",
      "__SCORECARD_BADGE_CLASS__": "badge-pass" if t.passed_count == t.total_checks else "badge-fail",
      "__PASSED_COUNT__": str(t.passed_count),
      "__TOTAL_CHECKS__": str(t.total_checks),
      "__ARTIFACT_CHIPS__": artifact_chips,
      "__KPI_CARDS__": _render_kpi_cards(t),
      "__CR_TEXT__": f"{t.compression_ratio:.2f}:1",
      "__MATERIAL_SWATCHES__": _render_material_swatches(t.material_swatches),
      "__IMEP_TEXT__": f"{imep_bar:.2f}",
      "__PEAK_HP_TEXT__": peak_hp_txt,
      "__PEAK_TQ_TEXT__": peak_tq_txt,
      "__DYNO_MECH_BADGE__": mech_badge,
      "__DYNO_OIL_SHIFT__": oil_shift,
      "__DYNO_WAV_B64__": t.dyno_wav_b64,
      "__MIN_BSFC_TEXT__": min_bsfc_txt,
      "__TOTAL_FMEP_TEXT__": f"{_fval(lub_fric, 'total_fmep_bar'):.2f} bar ({_fval(lub_fric, 'total_friction_power_w'):.0f} W)",
      "__LUB_SAFE_CLASS__": "badge-pass" if lub_safe else "badge-fail",
      "__LUB_SAFE_TEXT__": "KNOCK & THERMAL SAFE" if lub_safe else "CHECK THERMAL/KNOCK",
      "__LUB_ROWS__": _render_lub_rows(t),
      "__SCORECARD_ROWS__": _render_scorecard_rows(t.scorecard),
      "__DFM_BADGE_CLASS__": "badge-pass" if dfm_passed else "badge-fail",
      "__DFM_BADGE_TEXT__": dfm_mode_txt if dfm_passed else "DFM VIOLATION",
      "__DFM_ROWS__": _render_dfm_rows(t.dfm),
      "__BOM_TOTAL_TEXT__": f"{_fval(bom_val, 'total_cost_usd'):,.2f}",
      "__BOM_CHANNEL_PILLS__": _render_bom_channel_pills(t.bom),
      "__SCRIPT__": script_body,
  }
  doc = HTML_DOC_TEMPLATE
  for token, value in html_replacements.items():
    doc = doc.replace(token, value)
  return doc
