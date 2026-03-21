"""
Setup script for Tennis Tagger
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="tennis-tagger",
    version="1.0.0",
    author="Tennis Tagger Development Team",
    description="Automated tennis match tagging system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "opencv-python>=4.8.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "ultralytics>=8.0.0",
        "mediapipe>=0.10.7",
        "scikit-learn>=1.3.0",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
        "PyQt5>=5.15.9",
    ],
    entry_points={
        "console_scripts": [
            "tennis-tagger=main:main",
            "tennis-tagger-gui=src.gui:main",
            "tennis-tagger-train=scripts.train_custom:main",
        ],
    },
)
