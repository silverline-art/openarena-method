from setuptools import setup, find_packages

setup(
    name="exmo-gait-analyzer",
    version="1.0.0",
    description="Production-grade 3-view 120 FPS mouse gait analysis pipeline",
    author="EXMO Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "openpyxl>=3.1.0",
        "pyyaml>=6.0",
        "scikit-learn>=1.3.0",
        "pydantic>=2.0.0",
    ],
    entry_points={
        "console_scripts": [
            "exmo_gait_analyzer=exmo_gait.cli:main",
        ],
    },
    python_requires=">=3.8",
)
