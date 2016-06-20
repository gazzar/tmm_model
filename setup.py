from setuptools import setup
import re


version_file_contents = open('acsemble/version.py', 'r').read()
version = re.search(r"__version__ = '(.*)'", version_file_contents).group(1)
name = re.search(r"__name__ = '(.*)'", version_file_contents).group(1)

setup(
    name=name,
    version=version,
    packages=['acsemble'],
    entry_points={
        'console_scripts': [
            'main-script = acsemble.main:main',
        ]
    },
    install_requires=[
        'six==1.10.0',
        'click==6.6',
        'ruamel.yaml==0.11.11',
        'recipy>=0.2.3',
        'profilehooks==1.8.1',
        'imageio==1.5',
        'progressbar2==3.7.0'
    ],
)