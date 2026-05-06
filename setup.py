from setuptools import setup, find_packages
from pathlib import Path


requirements = Path("requirements.txt").read_text().splitlines()

setup(
    name="sciflow",                         
    version="0.0.1",                           
    packages=find_packages(where="src"),     
    package_dir={"": "src"},                   
    install_requires=requirements,
    classifiers=[                             
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
