"""
Setup script for AuraNet
"""

from setuptools import setup, find_packages
import os

# Read README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="auranet",
    version="1.0.0",
    author="AuraNet Team",
    author_email="auranet@example.com",
    description="A Dual-Stream Forensic Network for Face Manipulation Detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/AuraNet",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Security",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.1.0",
            "pytest-cov>=3.0.0",
            "black>=22.6.0",
            "flake8>=4.0.0",
            "isort>=5.10.0",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "ipywidgets>=7.7.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "auranet-train=train:main",
            "auranet-evaluate=evaluate:main",
            "auranet-demo=demo:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.md"],
    },
    project_urls={
        "Bug Reports": "https://github.com/your-username/AuraNet/issues",
        "Source": "https://github.com/your-username/AuraNet",
        "Documentation": "https://your-username.github.io/AuraNet/",
    },
)
