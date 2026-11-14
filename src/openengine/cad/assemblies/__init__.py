"""Hierarchical NestWorks C500 CAD sub-assemblies and full digital twin."""

from openengine.cad.assemblies.c500_fixture_assemblies import (
    C500FixtureReport,
    FixtureCheckItem,
    generate_c500_fixture_plate,
    generate_c500_fixture_setup_compound,
    generate_c500_rotary_chuck,
    get_c500_setup_part,
    verify_all_c500_fixtures,
)
from openengine.cad.assemblies.crankcase_assembly import (
    CrankcaseAssemblyReport,
    build_crankcase_assembly_parts,
    generate_crankcase_assembly,
    verify_crankcase_assembly,
)
from openengine.cad.assemblies.dressed_engine_assembly import (
    DressedEngineReport,
    build_dressed_engine_layer_groups,
    build_dressed_engine_parts,
    generate_dressed_engine_assembly,
    verify_dressed_engine,
)
from openengine.cad.assemblies.head_assembly import (
    HeadAssemblyReport,
    build_head_assembly_parts,
    generate_head_assembly,
    verify_head_assembly,
)
from openengine.cad.assemblies.rotating_assembly import (
    RotatingAssemblyReport,
    build_rotating_assembly_parts,
    generate_rotating_assembly,
    verify_rotating_assembly,
)

__all__ = [
    "C500FixtureReport",
    "CrankcaseAssemblyReport",
    "DressedEngineReport",
    "FixtureCheckItem",
    "HeadAssemblyReport",
    "RotatingAssemblyReport",
    "build_crankcase_assembly_parts",
    "build_dressed_engine_layer_groups",
    "build_dressed_engine_parts",
    "build_head_assembly_parts",
    "build_rotating_assembly_parts",
    "generate_c500_fixture_plate",
    "generate_c500_fixture_setup_compound",
    "generate_c500_rotary_chuck",
    "generate_crankcase_assembly",
    "generate_dressed_engine_assembly",
    "generate_head_assembly",
    "generate_rotating_assembly",
    "get_c500_setup_part",
    "verify_all_c500_fixtures",
    "verify_crankcase_assembly",
    "verify_dressed_engine",
    "verify_head_assembly",
    "verify_rotating_assembly",
]
