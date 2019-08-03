import codecs
from os import path
from setuptools import setup, find_packages


def long_description():
    """Returns the content of the readme."""
    this_directory = path.abspath(path.dirname(__file__))
    with codecs.open(
        path.join(this_directory, "README.md"), encoding="utf-8"
    ) as f:
        return f.read()


setup(
    name="evidencegraph",
    version="0.3.0",
    description="Evidence graphs for parsing argumentation structure",
    long_description_content_type="text/markdown",
    long_description=long_description(),
    author="Andreas Peldszus",
    author_email="andreas.peldszus@posteo.de",
    url="https://github.com/peldszus/evidencegraph",
    include_package_data=True,
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Natural Language :: German",
        "Operating System :: POSIX",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 2 :: Only",
        "Topic :: Text Processing :: Linguistic",
    ],
    keywords="argument-mining argumentation-mining nlp discourse",
)
