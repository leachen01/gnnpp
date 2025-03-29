from setuptools import setup, find_packages

with open('/home/ltchen/requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='gnnpp',
    version='1.0',
    packages=find_packages(),  # Automatische Erkennung aller Pakete (Ordner mit __init__.py)
    description='improving graph neural network ensemble postprocessing for weather forecasting',
    #long_description=open('README.md').read(),  # Längere Beschreibung aus der README.md Datei
    #long_description_content_type="text/markdown",  # Format der langen Beschreibung
    author='Lea Chen',
    url='https://github.com/leachen01/gnnpp',  # URL zu deinem Projekt-Repository
    install_requires=required,
    #classifiers=[
    #    "Programming Language :: Python :: 3",
    #    "License :: OSI Approved :: MIT License",
    #    "Operating System :: OS Independent",
    #],
)
