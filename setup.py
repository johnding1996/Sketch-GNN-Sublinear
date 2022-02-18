import setuptools
import os
import sys

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(here, 'sublinear_gnn'))

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sublinear_gnn",
    version='0.0.1',
    author="Mucong Ding",
    author_email="mcding@umd.edu",
    url="https://johnding1996.github.io/Sublinear-GNN/",
    description="Pre-Compression Approach to Sublinear GNNs.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
    ],
    license='MIT',
    packages=setuptools.find_packages(
        exclude=['datasets', 'results']),
    classifiers=[
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Intended Audience :: Science/Research',
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires='>=3.6',
)
