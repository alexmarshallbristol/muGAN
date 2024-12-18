from setuptools import setup

setup(
    name="SHiP_GAN_module",
    version="1.0",
    description="Muon background generation for SHiP experiments via GAN sampling",
    author="Alex Marshall",
    author_email="alex.marshall@cern.ch",
    packages=["SHiP_GAN_module"],  # same as name
    install_requires=["keras", "uproot"],  # external packages as dependencies
)
