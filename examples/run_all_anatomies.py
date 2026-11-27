"""Generate one sample from each anatomical profile.

Run: python examples/run_all_anatomies.py
"""
from pathlib import Path
from paforge.config import preset
from paforge.pipeline import run_batch
from paforge.io import plot_preview

root=Path("examples/demo_outputs")
for anatomy in ("forearm","skin","breast","carotid"):
    config=preset(anatomy)
    paths=run_batch(config,root)
    plot_preview(paths[0])
    print(f"{anatomy}: {paths[0]}")
