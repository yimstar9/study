import setuptools
from tistory.config.info import __version__, __author__, __contact__, __github__

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tistory",
    version=__version__,
    license='MIT',
    author=__author__,
    author_email=__contact__,
    description="Tistory Open API Python Wrapper",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=__github__,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
)
