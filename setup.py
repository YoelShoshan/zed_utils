import os
from setuptools import setup
from setuptools import find_packages

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "zed_utils",
    version = "0.0.1",
    author = "Yoel Shoshan",
    author_email = "yoelshoshan@gmail.com",
    description = ("Several utilities"),
    license = "BSD",
    url = "https://github.com/YoelShoshan/zed_utils",
    long_description=read('README.md'),
    packages=find_packages()
	)
	
#	    classifiers=[
#        "Development Status :: 3 - Alpha",
#       "Topic :: Utilities",
#        "License :: OSI Approved :: BSD License",
#    ],

#keywords = "example documentation tutorial",