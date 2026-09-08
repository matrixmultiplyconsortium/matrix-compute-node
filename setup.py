"""
Legacy setup.py configuration for matrix-compute-node.
Delegates configuration and metadata to pyproject.toml via setuptools>=61.0.
"""

from setuptools import setup, find_packages

setup(
    name="matrix-compute-node",
    version="0.1.0",
    packages=find_packages(where=".", include=["src*"]),
    package_dir={"": "."},
    entry_points={
        "console_scripts": [
            "matrix-compute-node=src.main:main",
        ],
    },
)
