#!/usr/bin/env python

import os
import sys

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

if sys.argv[-1] == "publish":
    os.system("python setup.py sdist upload")
    sys.exit()

readme = open("README.md").read()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="dms_3d_features",
    version="1.0.0",
    description="",
    long_description=readme,
    long_description_content_type="test/markdown",
    author="Joe Yesselman",
    author_email="jyesselm@unl.edu",
    url="",
    packages=[
        "dms_3d_features",
    ],
    package_dir={"dms_3d_features": "dms_3d_features"},
    py_modules=[
        "dms_3d_features/cli",
        "dms_3d_features/format_tables",
        "dms_3d_features/hbond",
        "dms_3d_features/logger",
        "dms_3d_features/paths",
        "dms_3d_features/pdb_features",
        "dms_3d_features/plotting",
        "dms_3d_features/process_motifs",
        "dms_3d_features/sasa",
        "dms_3d_features/stats",
    ],
    include_package_data=True,
    install_requires=requirements,
    zip_safe=False,
    keywords="dms_3d_features",
    classifiers=[
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
    entry_points={"console_scripts": []},
)
