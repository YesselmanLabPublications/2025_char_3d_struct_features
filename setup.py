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
    name="dms_quant_framework",
    version="1.0.0",
    description="code for the paper: a quantitative framework for structural interpretation of DMS reactivity",
    long_description=readme,
    long_description_content_type="test/markdown",
    author="Joe Yesselman",
    author_email="jyesselm@unl.edu",
    url="",
    packages=[
        "dms_quant_framework",
    ],
    package_dir={"dms_quant_framework": "dms_quant_framework"},
    py_modules=[
        "dms_quant_framework/cli",
        "dms_quant_framework/format_tables",
        "dms_quant_framework/hbond",
        "dms_quant_framework/logger",
        "dms_quant_framework/paths",
        "dms_quant_framework/pdb_features",
        "dms_quant_framework/plotting",
        "dms_quant_framework/process_motifs",
        "dms_quant_framework/sasa",
        "dms_quant_framework/stats",
    ],
    include_package_data=True,
    install_requires=requirements,
    zip_safe=False,
    keywords="dms_quant_framework",
    classifiers=[
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
    entry_points={"console_scripts": []},
)
