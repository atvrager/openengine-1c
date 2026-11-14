"""Static GitHub Pages bundle assembler and relative-link integrity verifier."""

from dataclasses import dataclass
from pathlib import Path
import re
import shutil
from typing import List, Tuple, Union

from openengine.reporting.dashboard import generate_interactive_dashboard
from openengine.reporting.summary_md import generate_markdown_summary

ZERO_INT: int = 0
EXTERNAL_SCHEMES: Tuple[str, ...] = ("http://", "https://", "data:", "mailto:", "javascript:", "#")
SCRIPT_BLOCK_RE = re.compile(r"<script\b[^>]*>.*?</script>", re.DOTALL | re.IGNORECASE)
HREF_ATTR_RE = re.compile(r'href="([^"]+)"')
ASSET_SUBDIRS: Tuple[Tuple[str, str], ...] = (
    ("reports/models", "models"),
    ("cad", "cad"),
    ("dyno", "dyno"),
    ("dfm/drawings", "dfm/drawings"),
    ("bom", "bom"),
    ("lubrication", "lubrication"),
)


@dataclass(frozen=True, slots=True)
class PagesSiteManifest:
  """Summary of assembled static GitHub Pages output directory and verified links."""

  site_dir: str
  index_html: str
  copied_file_count: int
  verified_links: Tuple[str, ...]


def _copy_dir_files(src_dir: Path, dst_dir: Path, pattern: str = "*") -> int:
  """Copy regular files matching pattern from src_dir to dst_dir and return count."""
  if not src_dir.exists() or not src_dir.is_dir():
    return ZERO_INT
  copied = ZERO_INT
  for src_file in sorted(src_dir.glob(pattern)):
    if src_file.is_file():
      dst_dir.mkdir(parents=True, exist_ok=True)
      shutil.copy2(src_file, dst_dir / src_file.name)
      copied += 1
  return copied


def _copy_pages_assets(out_root: Path, site_root: Path) -> int:
  """Copy generated CAD, dyno, DFM, setup-sheet, BoM, lubrication, and model assets."""
  copied_total = ZERO_INT
  for src_rel, dst_rel in ASSET_SUBDIRS:
    pattern = "*.step" if src_rel == "cad" else "*"
    copied_total += _copy_dir_files(out_root / src_rel, site_root / dst_rel, pattern=pattern)

  dfm_src = out_root / "dfm" / "c500_dfm_report.json"
  if dfm_src.is_file():
    (site_root / "dfm").mkdir(parents=True, exist_ok=True)
    shutil.copy2(dfm_src, site_root / "dfm" / "c500_dfm_report.json")
    copied_total += 1
  return copied_total


def _verify_relative_links(html_text: str, site_root: Path) -> Tuple[str, ...]:
  """Verify every non-external href inside index.html resolves inside site_root."""
  html_no_scripts = SCRIPT_BLOCK_RE.sub("", html_text)
  site_resolved = site_root.resolve()
  verified: List[str] = []
  seen: set[str] = set()

  for href in HREF_ATTR_RE.findall(html_no_scripts):
    if not href or href.startswith(EXTERNAL_SCHEMES) or href in seen:
      continue
    seen.add(href)
    target = (site_resolved / href).resolve()
    if site_resolved not in target.parents and target != site_resolved:
      raise ValueError(f"Relative asset link escapes site directory: {href}")
    if not target.is_file() or target.stat().st_size <= ZERO_INT:
      raise FileNotFoundError(f"Broken relative Pages asset link: {href} -> {target}")
    verified.append(href)
  return tuple(verified)


def build_pages_site(
    outputs_dir: Union[str, Path] = "outputs",
    site_dir: Union[str, Path] = "_site",
) -> PagesSiteManifest:
  """Assemble a self-contained static GitHub Pages bundle and verify local links."""
  out_root = Path(outputs_dir)
  site_root = Path(site_dir)
  site_root.mkdir(parents=True, exist_ok=True)
  (site_root / ".nojekyll").write_text("", encoding="utf-8")

  copied_count = _copy_pages_assets(out_root=out_root, site_root=site_root)
  summary_target = site_root / "summary.md"
  atlas_json = out_root / "fea" / "atlas" / "fea_atlas_results.json"
  generate_markdown_summary(
      output_md_path=summary_target,
      cht_json=out_root / "cfd" / "cht_cooling_results.json",
      closed_loop_json=out_root / "calibration" / "closed_loop_results.json",
      piston_fea_json=out_root / "fea" / "piston" / "piston_fea_results.json",
      conrod_fea_json=out_root / "fea" / "conrod" / "conrod_fea_results.json",
      gasket_fea_json=out_root / "fea" / "gasket" / "head_gasket_fea_results.json",
      fea_atlas_json=atlas_json,
      dfm_report_json=site_root / "dfm" / "c500_dfm_report.json",
      dyno_json=site_root / "dyno" / "dyno_results.json",
      bom_json=site_root / "bom" / "bom_summary.json",
      thermo_lub_json=site_root / "lubrication" / "thermo_lub_results.json",
      setup_sheets_json=site_root / "dfm" / "drawings" / "setup_sheets_manifest.json",
  )
  copied_count += 1

  index_path = site_root / "index.html"
  generate_interactive_dashboard(
      output_html_path=index_path,
      models_dir=site_root / "models",
      cht_json=out_root / "cfd" / "cht_cooling_results.json",
      closed_loop_json=out_root / "calibration" / "closed_loop_results.json",
      piston_fea_json=out_root / "fea" / "piston" / "piston_fea_results.json",
      conrod_fea_json=out_root / "fea" / "conrod" / "conrod_fea_results.json",
      gasket_fea_json=out_root / "fea" / "gasket" / "head_gasket_fea_results.json",
      fea_atlas_json=atlas_json,
      dfm_report_json=site_root / "dfm" / "c500_dfm_report.json",
      dyno_json=site_root / "dyno" / "dyno_results.json",
      dyno_wav=site_root / "dyno" / "exhaust_pull_44k1.wav",
      bom_json=site_root / "bom" / "bom_summary.json",
      thermo_lub_json=site_root / "lubrication" / "thermo_lub_results.json",
      setup_sheets_json=site_root / "dfm" / "drawings" / "setup_sheets_manifest.json",
      asset_url_prefix=".",
  )
  shutil.copy2(index_path, site_root / "engineering_dashboard.html")
  verified = _verify_relative_links(index_path.read_text(encoding="utf-8"), site_root)
  return PagesSiteManifest(
      site_dir=str(site_root),
      index_html=str(index_path),
      copied_file_count=copied_count,
      verified_links=verified,
  )
