from distutils.core import setup

with open("requirements.txt") as f:
    required_packages = f.read().splitlines()

setup(
    name="chaincodes",
    version="1.0",
    description="Transformations on the freeman chaincodes",
    author="Jay Paul Morgan",
    author_email="jay.morgan@univ-tln.fr",
    url="https://github.com/jaypmorgan/chaincodes",
    packages=["chaincodes"],
    install_requires=required_packages)
