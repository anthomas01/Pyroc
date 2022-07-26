from setuptools import setup, find_packages
import re
import os

__version__ = re.findall(
    r"""__version__ = ["']+([0-9\.]*)["']+""",
    open("pyroc/__init__.py").read(),
)[0]

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="pyroc",
    version=__version__,
    description="Cyclone Rocketry Python Environment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="Rocketry, Iowa State University",
    author="",
    author_email="",
    url="",
    license="",
    packages=find_packages(include=["pyroc*"]),
    install_requires=["numpy>=1.16", "scipy>=1.2"],
    extras_require={
        "testing": ["numpy-stl"],
    },
    classifiers=["Operating System :: OS Independent", "Programming Language :: Python"],
)