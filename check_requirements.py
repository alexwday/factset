#!/usr/bin/env python3
"""
Validate installed packages match requirements.txt version pins.
Run this after pulling code to ensure environment compatibility.
"""

import sys
import subprocess
import re
from pathlib import Path


def parse_requirements(requirements_path):
    """Parse requirements.txt and extract package names and pinned versions."""
    pinned_packages = {}

    with open(requirements_path, 'r') as f:
        for line in f:
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue

            # Parse package==version format
            if '==' in line:
                # Remove inline comments
                package_spec = line.split('#')[0].strip()
                package_name, version = package_spec.split('==', 1)
                pinned_packages[package_name.strip()] = version.strip()

    return pinned_packages


def get_installed_version(package_name):
    """Get installed version of a package using pip show."""
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'show', package_name],
            capture_output=True,
            text=True,
            check=False
        )

        if result.returncode != 0:
            return None

        # Parse version from pip show output
        for line in result.stdout.split('\n'):
            if line.startswith('Version:'):
                return line.split(':', 1)[1].strip()

        return None
    except Exception as e:
        print(f"Error checking {package_name}: {e}")
        return None


def main():
    """Main validation function."""
    script_dir = Path(__file__).parent
    requirements_file = script_dir / 'requirements.txt'

    if not requirements_file.exists():
        print(f"❌ ERROR: requirements.txt not found at {requirements_file}")
        sys.exit(1)

    print("🔍 Checking installed packages against requirements.txt...\n")

    pinned_packages = parse_requirements(requirements_file)

    if not pinned_packages:
        print("⚠️  WARNING: No version-pinned packages found in requirements.txt")
        sys.exit(0)

    mismatches = []
    missing = []
    matches = []

    for package_name, required_version in pinned_packages.items():
        installed_version = get_installed_version(package_name)

        if installed_version is None:
            missing.append(package_name)
            print(f"❌ {package_name}: NOT INSTALLED (required: {required_version})")
        elif installed_version != required_version:
            mismatches.append((package_name, installed_version, required_version))
            print(f"⚠️  {package_name}: {installed_version} (required: {required_version})")
        else:
            matches.append(package_name)
            print(f"✅ {package_name}: {installed_version}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY:")
    print(f"  ✅ Matching versions: {len(matches)}")
    print(f"  ⚠️  Version mismatches: {len(mismatches)}")
    print(f"  ❌ Missing packages: {len(missing)}")
    print("="*60)

    if mismatches or missing:
        print("\n🔧 RECOMMENDED ACTION:")
        print("   Run: pip install -r requirements.txt --upgrade")
        print("   Or:  pip install -r requirements.txt --force-reinstall")
        sys.exit(1)
    else:
        print("\n✨ All pinned packages match requirements.txt!")
        sys.exit(0)


if __name__ == '__main__':
    main()
