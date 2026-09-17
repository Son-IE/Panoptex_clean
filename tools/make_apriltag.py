#!/usr/bin/env python3
"""
make_apriltag.py -- generate printable tag36h11 AprilTags at an EXACT size.

    python3 tools/make_apriltag.py --ids 1 2 3 --size-mm 115 --out tags/
    python3 tools/make_apriltag.py --ids 0     --size-mm 150 --out tags/

Writes one SVG per tag. SVG carries real physical units, so printing at
100% / "Actual size" (NOT "Fit to page", which silently rescales) gives a tag
whose black square measures exactly --size-mm across.

WHY THIS MATTERS: `size-mm` is the outer edge of the BLACK SQUARE, which is
the same quantity as `tag_size_m` / `tag0_size_m` in config/floor_tags.yaml.
Every calibration node back-projects from that number, so an error there is a
pure scale error on the solve that no camera pose can absorb -- and it is
invisible in the imagery, since a wrong-sized tag detects perfectly well. The
printed caption records the intended size so a tag can always be checked
against a ruler later.

Tags 1/2/3 go flat on the floor; tag 0 rides on the robot. Tag 0 is the one
that has to stay decodable from across the room, so print it LARGER -- it
needs ~40 px of edge in the overhead image to be reliable, and it is also the
only tag that suffers motion blur. If tag 0's size differs from the floor
tags', record it as `tag0_size_m`.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

# tag36h11 carries 6x6 data cells inside a 1-cell black border -> the black
# square is 8 cells across. cv2 generates exactly that region, so mapping the
# image to --size-mm makes --size-mm the black square's outer edge.
BORDER_BITS = 1
CELLS = 8


def make_svg(tag_id: int, size_mm: float, quiet_cells: float) -> str:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    img = cv2.aruco.generateImageMarker(dictionary, tag_id, CELLS, BORDER_BITS)
    if img.shape != (CELLS, CELLS):
        raise RuntimeError(f"unexpected marker shape {img.shape}")

    cell = size_mm / CELLS
    quiet = quiet_cells * cell
    page = size_mm + 2 * quiet
    caption_h = 10.0

    rects = []
    for row in range(CELLS):
        for col in range(CELLS):
            if img[row, col] == 0:                      # black cell
                rects.append(
                    f'<rect x="{quiet + col * cell:.4f}" '
                    f'y="{quiet + row * cell:.4f}" '
                    f'width="{cell:.4f}" height="{cell:.4f}" fill="#000"/>')

    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     width="{page:.4f}mm" height="{page + caption_h:.4f}mm"
     viewBox="0 0 {page:.4f} {page + caption_h:.4f}">
  <rect width="100%" height="100%" fill="#fff"/>
{chr(10).join("  " + r for r in rects)}
  <!-- ruler check: this line spans exactly the black square -->
  <text x="{page / 2:.4f}" y="{page + 6:.4f}" font-family="monospace"
        font-size="4" text-anchor="middle" fill="#000">
    tag36h11  id={tag_id}  black square = {size_mm:.1f} mm  (print at 100%)
  </text>
</svg>
'''


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", type=int, nargs="+", required=True)
    ap.add_argument("--size-mm", type=float, required=True,
                    help="outer edge of the BLACK SQUARE, millimetres")
    ap.add_argument("--quiet-cells", type=float, default=1.0,
                    help="white margin around the tag, in cells (>=1 needed)")
    ap.add_argument("--out", default=".", help="output directory")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for tag_id in args.ids:
        path = out / f"tag36h11_{tag_id:02d}_{args.size_mm:.0f}mm.svg"
        path.write_text(make_svg(tag_id, args.size_mm, args.quiet_cells))
        print(f"wrote {path}")

    print("\nPrint at 100% / 'Actual size' -- NOT 'Fit to page'.")
    print("Then MEASURE the black square with a ruler and put that number, in")
    print("metres, into config/floor_tags.yaml (tag_size_m / tag0_size_m).")


if __name__ == "__main__":
    main()
