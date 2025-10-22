from setuptools import setup, find_packages
import os

# Read requirements
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="FrizzAnalysis",
    version="1.0.0",
    description="Hair Frizz Analysis Tool for quantitative frizz testing",
    author="Your Name",
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'frizz-analysis=src.gui:main',
        ],
        'gui_scripts': [
            'frizz-analysis-gui=src.gui:main',
        ]
    },
    include_package_data=True,
    package_data={
        '': ['icons/*.ico', 'icons/*.icns', 'icons/*.png'],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
