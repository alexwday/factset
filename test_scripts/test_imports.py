#!/usr/bin/env python3
"""
Test script to check what FactSet SDK modules are available
"""

print("Testing FactSet SDK imports...")

try:
    import fds.sdk.EventsandTranscripts
    print("✅ fds.sdk.EventsandTranscripts - Available")
except ImportError as e:
    print(f"❌ fds.sdk.EventsandTranscripts - Not available: {e}")

try:
    import fds.sdk.StreetAccountNews
    print("✅ fds.sdk.StreetAccountNews - Available")
except ImportError as e:
    print(f"❌ fds.sdk.StreetAccountNews - Not available: {e}")

try:
    from smb.SMBConnection import SMBConnection
    print("✅ smb.SMBConnection - Available")
except ImportError as e:
    print(f"❌ smb.SMBConnection - Not available: {e}")

try:
    import pandas as pd
    print("✅ pandas - Available")
except ImportError as e:
    print(f"❌ pandas - Not available: {e}")

try:
    from dotenv import load_dotenv
    print("✅ python-dotenv - Available")
except ImportError as e:
    print(f"❌ python-dotenv - Not available: {e}")

print("\nPython path:")
import sys
for path in sys.path:
    print(f"  {path}")