# OpenEngine-1C

OpenEngine-1C is a code-first engineering design, simulation, and verification pipeline for a single-cylinder internal combustion engine.

The baseline engine uses a 196.1 cm3 four-stroke SOHC two-valve architecture (68 mm bore x 54 mm stroke, 9.0:1 compression ratio).

## Documentation
* See [docs/openengine_1c_master_plan.md](docs/openengine_1c_master_plan.md) for the multi-physics design specification and verification targets.
* See [docs/openengine_1c_dfm_plan.md](docs/openengine_1c_dfm_plan.md) for the manufacturability plan, fuel derating, wet-sump splash cooling, and CNC setup strategy.

## Engineering Capabilities & Stack
* **Language & Package Manager:** Python 3.11 or newer managed with `uv`.
* **0D/1D Thermodynamics & Fuel Kinetics:** `Cantera` models Wiebe combustion, Woschni heat transfer, and US 87-AKI E10 knock margins.
* **Lubrication & Dynamics:** `NumPy`, `SciPy`, and `SymPy` solve 4-zone FMEP, wet-sump splash heat balance, slider-crank forces, and SOHC polynomial cam kinematics.
* **Parametric 3D CAD & Styling:** `build123d` on Open CASCADE exports healed B-Rep STEP assemblies with XCAF alloy colors and opacities for all 116 dressed solids.
* **DFM & Setup Drawings:** NestWorks C500 CNC B-Rep checks audit machinability, fallback 4th-axis parts, and generate 10 2D SVG/DXF shop setup sheets.
* **Costed Bill of Materials:** The BoM module resolves a 40-item McMaster-Carr and commercial powersports parts list with machine-time rollups.
* **Containerized Solvers (`podman`):**
  * **CFD:** OpenFOAM v2406 solves steady port flow (`simpleFoam`) and moving-mesh combustion (`XiEngineFoam`).
  * **Meshing & FEA:** `Gmsh` generates C3D10 tetrahedral volume meshes and `CalculiX` (`ccx`) computes structural and thermal stress.
* **Virtual Dyno & GitHub Pages Dashboard:** `openengine dyno` computes SAE J1349 torque curves and 44.1 kHz exhaust audio. `openengine build-pages` builds a Three.js and Plotly dashboard bundle.

## Quickstart

### 1. Set Up Python Environment
```bash
uv sync
```

### 2. Build Solver Container
```bash
podman build -t openengine-solvers:latest -f Containerfile .
```

### 3. Run Pipeline & Build Static Site
```bash
uv run openengine run-all --quick --output-dir outputs
uv run openengine build-pages --outputs-dir outputs --site-dir _site
```

### 4. Run Test Suite
```bash
uv run pytest
```
