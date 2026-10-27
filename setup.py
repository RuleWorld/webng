from setuptools import setup, find_packages
from webng.core.version import get_version

VERSION = get_version()

f = open("README.md", "r")
LONG_DESCRIPTION = f.read()
f.close()

# Minimum versions are the oldest seen working with this WEBNG: older WESTPA
# (e.g. 2022.12) breaks the executable propagator's pcoord loader and
# per-iteration bin mapper records, and libroadrunner before 2.8 can fail to
# load under NumPy 2.
INSTALL_REQUIRES = [
    "numpy",
    "cement",
    "pyyaml",
    "colorlog",
    "seaborn",
    "h5py",
    "pygpcca",
    "bionetgen>=0.8.7",
    "libroadrunner>=2.8.0",
    "networkx",
    "westpa>=2022.15",
    "scikit-learn",
    "nbformat"
]

EXTRAS_REQUIRE = {
    "dev" : ['pytest', 'twine>=1.11.0', 'wheel>=0.31.0']
}

setup(
    name="webng",
    version=VERSION,
    description="A weighted ensemble simulation setup tool starting from a BNGL model",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="Ali Sinan Saglam",
    author_email="asinansaglam@gmail.com",
    url="https://github.com/ASinanSaglam/BNG_WESTPA_pipeline",
    license="unlicensed",
    packages=find_packages(exclude=["ez_setup", "tests*"]),
    package_data={"webng": ["templates/*", "assets/*"]},
    include_package_data=True,
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    entry_points="""
        [console_scripts]
        webng = webng.main:main
    """,
)