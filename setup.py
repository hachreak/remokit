"""setuptools."""

from setuptools import setup  # , find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))
# packages = find_packages('src', exclude=['contrib', 'docs', 'tests'])

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='remokit',
    version='0.1.0',
    description='Ant game!',
    url='https://github.com/hachreak/remokit',
    author='Leonardo Rossi',
    author_email='leonardo.rossi@studenti.unipr.it',
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],

    keywords='machine learning',
    packages={'': 'src'},  # packages,  # Required
    install_requires=[
        'numpy>=1.14.5',
        'Keras>=2.2.0',
        'tensorflow>=1.9.0',
        'dlib>=19.15.0',
        'scikit-learn>=0.19.2',
    ],
    extras_require={  # Optional
        'dev': ['check-manifest'],
        'test': ['coverage'],
    }
)
