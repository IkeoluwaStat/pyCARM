__authors__ = 'Ogendengbe Ikeoluwa Ireoluwa, R. H. Liem'

__date__ = '17th July, 2025'

import setuptools
from os.path import exists, join

def readme():
    try:
        with open('README.md') as f:
            return f.read()
    except IOError:
        return ''

dependencies = [
    numpy,
    matplotlib,
    distinctipy,
]

setuptools.setup(
    name = "pyCARM",
    version = "1.0.0",
    author = "The Hong Kong University of Science",
    author_email = "iiogedengbe@connect.ust.hk",
    description = "pyCARM: Cellular Automata for Aircraft Arrival Modelling",
    long_description = readme(),
    long_description_content_type = "text/markdown",
    ###
    url = "https://github.com/IkeoluwaSta/pyCARM",
    project_urls = {
        "Source Code": "https://github.com/IkeoluwaSta/pyCARM",
        "Bug Tracker": "https://github.com/IkeoluwaSta/pyCARM/issues",
    },
    install_requires=dependencies,

    python_requires = ">=3",
    
    classifiers = [
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
          
    ],
    packages=['pyCARM'],
    include_package_data=True,
    package_data={'pyCARM': ['data/*']}, 
)
