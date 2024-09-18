"""Package installation setup."""

from setuptools import find_packages, setup

setup(
    name="uniception",
    version="0.1.0",
    description="Generalizable Perception Stack",
    author="AirLab",
    license="BSD Clause-3",
    packages=find_packages(),
    package_dir={"": "."},
    include_package_data=True,
)
