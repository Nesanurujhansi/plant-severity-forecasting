"""
Generate licence-safe synthetic leaf sample images for Gradio UI examples.
"""

from pathlib import Path
from PIL import Image, ImageDraw

examples_dir = Path("app/ui/examples")
examples_dir.mkdir(parents=True, exist_ok=True)

# Sample 1: Green leaf with 10% lesion spot
img1 = Image.new("RGB", (300, 300), color=(240, 240, 240))
draw1 = ImageDraw.Draw(img1)
draw1.polygon([(150, 30), (260, 100), (230, 240), (150, 280), (70, 240), (40, 100)], fill=(45, 140, 45), outline=(30, 90, 30))
draw1.ellipse([(120, 120), (170, 170)], fill=(130, 70, 30))
img1.save(examples_dir / "sample_leaf_01.png", format="PNG")

# Sample 2: Green leaf with multiple spots
img2 = Image.new("RGB", (300, 300), color=(245, 245, 245))
draw2 = ImageDraw.Draw(img2)
draw2.polygon([(150, 25), (270, 90), (240, 250), (150, 285), (60, 250), (30, 90)], fill=(50, 150, 50), outline=(35, 100, 35))
draw2.ellipse([(90, 90), (120, 120)], fill=(140, 60, 25))
draw2.ellipse([(160, 150), (200, 190)], fill=(140, 60, 25))
draw2.ellipse([(110, 200), (140, 230)], fill=(140, 60, 25))
img2.save(examples_dir / "sample_leaf_02.png", format="PNG")

# Write README.md
readme_content = (
    "# Gradio UI Example Images\n\n"
    "This directory contains procedurally generated synthetic leaf images used as Gradio UI example inputs.\n\n"
    "- `sample_leaf_01.png`: Synthetic green wheat leaf with a single moderate Septoria lesion spot.\n"
    "- `sample_leaf_02.png`: Synthetic green wheat leaf with multiple rust lesion spots.\n\n"
    "**Licence**: CC-BY-4.0 (Redistributable and free for demonstration and testing purposes).\n"
)
with open(examples_dir / "README.md", "w", encoding="utf-8") as f:
    f.write(readme_content)

print("[INFO] Gradio example images and README successfully created.")
