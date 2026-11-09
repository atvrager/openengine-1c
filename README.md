# OpenEngine-1C

A 100% open-source, code-first engineering design, thermodynamic simulation, and CAD/CFD/FEA verification pipeline for a single-cylinder, four-stroke SOHC internal combustion engine.

## Documentation
* See [docs/openengine_1c_master_plan.md](docs/openengine_1c_master_plan.md) for the full architectural specification and multi-physics verification roadmap.

## Toolchain & Stack
* **Language & Package Manager:** Python >= 3.11 managed via `uv`
* **0D/1D Thermodynamics & Chemistry:** `Cantera` (Wiebe combustion + Woschni heat transfer)
* **Mechanics & Dynamics:** `NumPy` / `SciPy` / `SymPy` (Slider-crank balancer, flywheel sizing, 4-5-6-7 polynomial cam profile)
* **Parametric 3D CAD:** `build123d` on Open CASCADE Technology (OCCT)
* **Containerized Solvers (`podman`):**
  * **CFD:** OpenFOAM v2406 (Steady port flow bench via `simpleFoam`, 3D moving-mesh in-cylinder combustion via `XiEngineFoam`)
  * **Meshing:** `Gmsh` (B-Rep to C3D10 solid tetrahedral volume mesh) & `cfMesh`
  * **FEA:** `CalculiX` (`ccx` for thermal-mechanical stress & column buckling)

## Quickstart

### 1. Python Environment Setup
```bash
# Sync dependencies and lockfile
uv sync
```

### 2. Containerized Solvers (Podman)
```bash
# Build the solver image
podman build -t openengine-solvers:latest -f Containerfile .
```

### 3. Run Tests
```bash
uv run pytest
```
