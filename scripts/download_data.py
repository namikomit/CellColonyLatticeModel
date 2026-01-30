#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download simulation data from ERDA share.

Downloads files from https://sid.erda.dk/sharelink/l8bcd32JVM
specifically the contents of simulation_data/no_pushing/
"""

import os
import sys
import urllib.request
import subprocess
from pathlib import Path

# Base URL for direct downloads
BASE_URL = "https://sid.erda.dk/share_redirect/l8bcd32JVM/simulation_data/no_pushing"

# Files available in simulation_data/no_pushing/
FILES = [
    "2drop_CRE11001000_rp1_icc75000_g0.025_ka2.3_A0_0.rar",
    "2drop_CRE11001000_rp1_icc75000_g0.025_ka2.3_A3_0.rar",
    "2drop_CRE11001000_rp1_icc75000_g0.025_ka2.3_A4_0.rar",
    "2drop_CRE11001000_rp1_icc75000_g0.025_ka2.3_A5_0.rar",
    "2drop_CRE11001000_rp1_icc75000_g0.025_ka2.3_A6_0.rar",
    "2drop_CRE11001000_rp1_icc75000_g0.025_ka2.3_A7_1.rar",
]


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def download_file(url: str, dest: Path) -> bool:
    """
    Download a file from URL to destination.

    Returns True if successful, False otherwise.
    """
    try:
        print(f"Downloading {dest.name}...")
        urllib.request.urlretrieve(url, dest)
        print(f"  Downloaded: {dest.name} ({dest.stat().st_size / 1024 / 1024:.1f} MB)")
        return True
    except Exception as e:
        print(f"  Error downloading {dest.name}: {e}")
        return False


def extract_rar(rar_path: Path, extract_to: Path) -> bool:
    """
    Extract a RAR file to the specified directory.

    Tries unrar command first, then rarfile Python package.
    Returns True if successful, False otherwise.
    """
    try:
        # Try using unrar command
        result = subprocess.run(
            ["unrar", "x", "-o-", str(rar_path), str(extract_to) + "/"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"  Extracted: {rar_path.name}")
            return True
        else:
            raise FileNotFoundError("unrar failed")
    except FileNotFoundError:
        # Try using rarfile Python package
        try:
            import rarfile
            with rarfile.RarFile(rar_path) as rf:
                rf.extractall(extract_to)
            print(f"  Extracted: {rar_path.name}")
            return True
        except ImportError:
            print(f"  Warning: Cannot extract {rar_path.name}")
            print("    Install unrar (apt install unrar) or rarfile (pip install rarfile)")
            return False
        except Exception as e:
            print(f"  Error extracting {rar_path.name}: {e}")
            return False


def main():
    """Main function to download and optionally extract data files."""
    project_root = get_project_root()
    data_dir = project_root / "data"

    # Create data directory if it doesn't exist
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Project root: {project_root}")
    print(f"Data directory: {data_dir}")
    print()

    # Parse command line arguments
    extract = "--extract" in sys.argv
    force = "--force" in sys.argv

    if "--help" in sys.argv or "-h" in sys.argv:
        print("Usage: python download_data.py [OPTIONS]")
        print()
        print("Options:")
        print("  --extract    Extract RAR files after downloading")
        print("  --force      Re-download files even if they exist")
        print("  --help, -h   Show this help message")
        return

    downloaded = 0
    skipped = 0
    extracted = 0

    for filename in FILES:
        url = f"{BASE_URL}/{filename}"
        dest = data_dir / filename

        # Check if file already exists
        if dest.exists() and not force:
            print(f"Skipping {filename} (already exists)")
            skipped += 1
        else:
            if download_file(url, dest):
                downloaded += 1

        # Extract if requested
        if extract and dest.exists():
            # Check if already extracted (look for folder with same base name)
            folder_name = filename.replace(".rar", "")
            extracted_folder = data_dir / folder_name

            if extracted_folder.exists() and not force:
                print(f"  Skipping extraction (folder exists): {folder_name}")
            else:
                if extract_rar(dest, data_dir):
                    extracted += 1

    print()
    print(f"Summary: {downloaded} downloaded, {skipped} skipped, {extracted} extracted")


if __name__ == "__main__":
    main()
