import os

from setuptools import find_packages, setup

setup(
    name="gym_envs",
    author="Yi Liu",
    author_email="yiyiliu.liu@ugent.be",
    packages=find_packages(),
    include_package_data=True,
    install_requires=["gym>=0.22, <=0.23", "gym-robotics", "numpy", "scipy"],
    extras_require={
        "tests": ["pytest-cov"],
        "codestyle": ["black", "isort", "pytype"],
        "docs": ["sphinx", "sphinx-rtd-theme"],
        "extra": ["numpngw", "stable-baselines3"],
    },
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
