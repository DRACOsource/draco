from setuptools import setup, find_packages

setup(
    name='v2x',
    version='0.1',
    author='j80055002',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'mesa'
    ]
)