"""Post-processing, interactive 3D WebGL dashboards, and multi-physics verification reports."""

from openengine.reporting.dashboard import (
    generate_interactive_dashboard,
    load_json_or_default,
)
from openengine.reporting.gltf_export import (
    export_cad_models_to_gltf,
    export_fea_stress_to_webgl,
    turbo_colormap,
)
from openengine.reporting.summary_md import generate_markdown_summary

__all__ = [
    "export_cad_models_to_gltf",
    "export_fea_stress_to_webgl",
    "generate_interactive_dashboard",
    "generate_markdown_summary",
    "load_json_or_default",
    "turbo_colormap",
]
