"""Quality contact sheet generator for static in-the-wild datasets.

Renders translucent mask overlays over source RGB images in grid layouts
and exports PNG contact sheets to reports/figures/ for human inspection.
"""

import sys
import argparse
from pathlib import Path
from typing import Any, Optional, List
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml.utils.paths import get_paths
from ml.utils.logging_utils import get_logger
from ml.utils.io import read_parquet

logger = get_logger(__name__)

def overlay_mask_on_image(
    image: Image.Image,
    lesion_mask: Optional[Image.Image] = None,
    leaf_mask: Optional[Image.Image] = None,
    alpha: float = 0.4
) -> Image.Image:
    """Overlay translucent lesion (red) and leaf (green) masks on RGB image."""
    img_rgb = image.convert("RGBA")
    w, h = img_rgb.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))

    if leaf_mask is not None:
        leaf_arr = np.array(leaf_mask.convert("L")) > 0
        leaf_color = np.zeros((h, w, 4), dtype=np.uint8)
        leaf_color[leaf_arr] = [0, 255, 0, int(255 * alpha * 0.5)] # Translucent green
        overlay = Image.alpha_composite(overlay, Image.fromarray(leaf_color))

    if lesion_mask is not None:
        lesion_arr = np.array(lesion_mask.convert("L")) > 0
        lesion_color = np.zeros((h, w, 4), dtype=np.uint8)
        lesion_color[lesion_arr] = [255, 0, 0, int(255 * alpha)] # Translucent red
        overlay = Image.alpha_composite(overlay, Image.fromarray(lesion_color))

    blended = Image.alpha_composite(img_rgb, overlay)
    return blended.convert("RGB")

def render_contact_sheet_grid(
    image_mask_tuples: List[Tuple[Image.Image, Optional[Image.Image], Optional[Image.Image]]],
    grid_cols: int = 5,
    cell_size: Tuple[int, int] = (150, 150)
) -> Image.Image:
    """Render a list of overlaid (image, lesion_mask, leaf_mask) tuples into a grid."""
    if not image_mask_tuples:
        return Image.new("RGB", cell_size, (200, 200, 200))

    n_items = len(image_mask_tuples)
    grid_rows = int(np.ceil(n_items / float(grid_cols)))

    grid_w = grid_cols * cell_size[0]
    grid_h = grid_rows * cell_size[1]
    grid_img = Image.new("RGB", (grid_w, grid_h), (240, 240, 240))

    for idx, (img, lesion_m, leaf_m) in enumerate(image_mask_tuples):
        col = idx % grid_cols
        row = idx // grid_cols

        overlaid = overlay_mask_on_image(img, lesion_m, leaf_m)
        cell_resized = overlaid.resize(cell_size, Image.Resampling.BILINEAR)

        x_offset = col * cell_size[0]
        y_offset = row * cell_size[1]
        grid_img.paste(cell_resized, (x_offset, y_offset))

    return grid_img

def generate_contact_sheets(
    manifest_df: pd.DataFrame,
    figures_dir: Optional[Path] = None,
    max_per_source: int = 50,
    seed: int = 42
) -> List[Path]:
    """Generate quality contact sheets per dataset source in reports/figures/."""
    paths = get_paths()
    if figures_dir is None:
        figures_dir = paths.figures_dir
    figures_dir.mkdir(parents=True, exist_ok=True)

    if manifest_df.empty:
        logger.warning("Empty manifest_df passed to generate_contact_sheets.")
        return []

    np.random.seed(seed)
    created_paths = []

    for source_name, group in manifest_df.groupby("source"):
        sample_rows = group.sample(n=min(len(group), max_per_source), random_state=seed).to_dict("records")
        tuples = []

        for row in sample_rows:
            try:
                img_p = paths.project_root / row["rel_image_path"]
                lesion_p = paths.project_root / row["rel_lesion_mask_path"]
                leaf_p = paths.project_root / row["rel_leaf_mask_path"] if row.get("rel_leaf_mask_path") else None

                img = Image.open(img_p)
                lesion_m = Image.open(lesion_p) if lesion_p.exists() else None
                leaf_m = Image.open(leaf_p) if leaf_p and leaf_p.exists() else None

                tuples.append((img, lesion_m, leaf_m))
            except Exception as e:
                logger.warning(f"Error loading images for contact sheet row {row.get('image_id')}: {e}")

        if tuples:
            grid_img = render_contact_sheet_grid(tuples, grid_cols=5, cell_size=(150, 150))
            out_file = figures_dir / f"static_contact_sheet_{source_name}.png"
            grid_img.save(out_file)
            created_paths.append(out_file)
            logger.info(f"Saved contact sheet for {source_name} ({len(tuples)} items) to {out_file}")

    return created_paths

def main() -> int:
    paths = get_paths()
    manifest_p = paths.project_root / "data" / "processed" / "static" / "manifest.parquet"
    if not manifest_p.exists():
        from ml.data.build_static_manifest import process_static_manifest
        manifest_df = process_static_manifest()
    else:
        manifest_df = read_parquet(manifest_p)

    generate_contact_sheets(manifest_df)
    return 0

if __name__ == "__main__":
    sys.exit(main())
