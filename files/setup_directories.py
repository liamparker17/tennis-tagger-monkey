"""
Setup Required Directories

Creates all necessary directories for the tennis tagger system.
Run this before first use or if you get directory-related errors.
"""

from pathlib import Path


def setup_directories():
    """Create all required directories"""
    directories = [
        # Core directories
        "logs",
        "data",
        "data/output",
        "data/training_pairs",
        "data/datasets",
        "data/feature_extraction",
        "cache",
        "models",
        "models/versions",

        # Config directory (should already exist)
        "config",
    ]

    for dir_path in directories:
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created/verified: {dir_path}")

    print("\n✅ All directories ready!")


if __name__ == "__main__":
    setup_directories()
