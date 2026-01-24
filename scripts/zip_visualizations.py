#!/usr/bin/env python3
"""
Zip all `visualization` folders under a pipeline outputs directory.

Usage:
    python scripts/zip_visualizations.py                      # zip all visualizations in pipeline_outputs
    python scripts/zip_visualizations.py --base /path/to/dir  # use custom base dir
    python scripts/zip_visualizations.py --group name         # zip only a specific group (folder name)
    python scripts/zip_visualizations.py --dir /path/to/visualization --single  # zip a single visualization dir

Creates: <group>/visualization.zip (or <group>_visualization.zip in base dir if --same-dir is not used)
"""

import argparse
import os
import zipfile
from pathlib import Path


def zip_dir(source_dir: Path, zip_path: Path, overwrite: bool = False) -> None:
    if zip_path.exists():
        if overwrite:
            zip_path.unlink()
        else:
            print(f"Skipping existing zip: {zip_path}")
            return

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(source_dir):
            for f in files:
                file_path = Path(root) / f
                # write with relative path inside zip
                zf.write(file_path, file_path.relative_to(source_dir))
    print(f"Created zip: {zip_path}")


def main():
    parser = argparse.ArgumentParser(description="Zip visualization folders in pipeline_outputs")
    parser.add_argument("--base", dest="base_dir", default="/openbayes/home/dynamic_stereo/pipeline_outputs",
                        help="Base pipeline outputs directory")
    parser.add_argument("--group", dest="group", default=None,
                        help="Only process a single group folder name (e.g. openbayes_home_sample_001_preds)")
    parser.add_argument("--dir", dest="single_dir", default=None,
                        help="Path to a single visualization folder to zip directly")
    parser.add_argument("--aggregate", action="store_true",
                        help="Create a single zip containing all visualization folders")
    parser.add_argument("--out", dest="out", default=None,
                        help="Output path for aggregate zip (used with --aggregate)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing zip files")
    parser.add_argument("--out-inside", action="store_true",
                        help="Place zip file inside the group folder as visualization.zip (default: create <group>_visualization.zip in base dir)")
    args = parser.parse_args()

    base = Path(args.base_dir)

    if args.single_dir:
        vis = Path(args.single_dir)
        if not vis.exists() or not vis.is_dir():
            print(f"Visualization folder not found: {vis}")
            return
        parent = vis.parent
        zip_name = parent / "visualization.zip" if args.out_inside else base / f"{parent.name}_visualization.zip"
        zip_dir(vis, zip_name, overwrite=args.overwrite)
        return

    if not base.exists() or not base.is_dir():
        print(f"Base directory not found: {base}")
        return

    groups = [p for p in base.iterdir() if p.is_dir()]
    if args.group:
        groups = [g for g in groups if g.name == args.group]
        if not groups:
            print(f"Group not found: {args.group}")
            return

    # If aggregate is requested, build a single zip containing each group's visualization
    if getattr(args, "aggregate", False):
        out_zip = Path(args.out) if getattr(args, "out", None) else base / "pipeline_visualizations.zip"
        if out_zip.exists() and not args.overwrite:
            print(f"Aggregate zip already exists: {out_zip} (use --overwrite to replace)")
            return

        with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            added_any = False
            for grp in sorted(groups):
                vis = grp / "visualization"
                if vis.exists() and vis.is_dir():
                    for root, _, files in os.walk(vis):
                        for f in files:
                            file_path = Path(root) / f
                            # arcname includes group and visualization path
                            rel = file_path.relative_to(vis)
                            arcname = Path(grp.name) / "visualization" / rel
                            zf.write(file_path, arcname)
                    added_any = True
                else:
                    print(f"No visualization folder in {grp} (skipping)")

        if added_any:
            print(f"Created aggregate zip: {out_zip}")
        else:
            print("No visualization folders found to aggregate")
        return

    for grp in sorted(groups):
        vis = grp / "visualization"
        if vis.exists() and vis.is_dir():
            if args.out_inside:
                zip_path = grp / "visualization.zip"
            else:
                zip_path = base / f"{grp.name}_visualization.zip"
            zip_dir(vis, zip_path, overwrite=args.overwrite)
        else:
            print(f"No visualization folder in {grp} (skipping)")


if __name__ == "__main__":
    main()
