import os
# read the contents of your README file
from pathlib import Path

import pkg_resources
from setuptools import find_packages, setup

long_description = Path(__file__).with_name("README.md").read_text()

setup(
    name='onnx_transformers',
    packages=find_packages(exclude=("test")),
    version='1.1.0',
    license='Apache Software License',
    description='tbd',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='c7nw3r',
    url='https://github.com/leftshiftone/onnx_transformers',
    download_url='https://github.com/leftshiftone/onnx_transformers/archive/refs/tags/v1.1.0.tar.gz',
    keywords=['onnx', 'transformers'],
    setup_requires=['setuptools_scm'],
    include_package_data=True,
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
    ]
)
