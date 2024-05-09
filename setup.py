from pathlib import Path
from setuptools import find_packages, setup

# Load version number
__version__ = ""
version_file = Path(__file__).parent.absolute() / "admet_ai" / "_version.py"

with open(version_file) as fd:
    exec(fd.read())

# Load README
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="admet_ai",
    version=__version__,
    author="Kyle Swanson",
    author_email="swansonk.14@gmail.com",
    description="admet_ai",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/swansonk14/admet_ai",
    download_url=f"https://github.com/swansonk14/admet_ai/archive/refs/tags/v_{__version__}.tar.gz",
    license="MIT",
    packages=find_packages(),
    package_data={"admet_ai": ["py.typed", "resources/**/*"]},
    entry_points={
        "console_scripts": [
            "admet_predict=admet_ai.admet_predict:admet_predict_command_line",
            "admet_web=admet_ai.web.run:admet_web",
        ]
    },
    install_requires=[
        "chemfunc>=1.0.4",
        "chemprop==1.6.1",
        "numpy",
        "pandas>=2.0.0,<2.2.0",  # remove this limit once rdkit implements a fix to PandasTools
        "rdkit>=2023.3.3",
        "seaborn",
        "tqdm",
        "typed-argument-parser>=1.9.0",
    ],
    extras_require={
        "tdc": ["openpyxl", "PyTDC>=0.4.1"],
        "web": ["flask", "gunicorn"],
    },
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Typing :: Typed",
    ],
    keywords=[
        "machine learning",
        "drug design",
        "ADMET",
        "molecular property prediction",
    ],
)
