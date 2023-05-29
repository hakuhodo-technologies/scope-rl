from setuptools import setup, find_packages
from os import path
import sys

from ofrl.version import __version__


here = path.abspath(path.dirname(__file__))
sys.path.insert(0, path.join(here, "scope_rl"))

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


# setup SCOPE-RL
setup(
    name="scope-rl", 
    version=__version__, 
    description="SCOPE-RL: A pipeline for offline reinforcement learning research and applications",
    url="https://github.com/negocia-inc/scope-rl",  # [TODO]
    author="Haruka Kiyohara",
    author_email="scope-rl@googlegroups.com",
    keywords=["off-policy evaluation", "offline reinforcement learning", "risk assessment"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "scipy>=1.10.1",
        "numpy==1.22.4",  # [TODO]
        "pandas>=1.5.3",
        "scikit-learn>=1.0.2",
        "matplotlib>=3.7.1",
        "seaborn==0.11.2",  # [TODO]
        "torch>=2.0.0",
        "d3rlpy>=1.1.1",
        "gym>=0.26.2",
        "gymnasium>=0.28.1",
        "hydra-core>=1.3.2",
    ],
    license="Apache License",
    packages=find_packages(
        exclude=[".github", "docs", "examples", "images", "tests"],
    ),
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: Apache Software License",
    ],
)
