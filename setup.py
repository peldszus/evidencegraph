from setuptools import setup, find_packages

setup(
    name="evidencegraph",
    version="0.3.0",
    description="Parsing argumentation structure ",
    long_description=open('README.md', 'rt').read(),
    author="Andreas Peldszus",
    url="https://github.com/peldszus/evidencegraph",
    include_package_data=True,
    package_dir={'': 'src'},
    packages=find_packages(where='src')
)
