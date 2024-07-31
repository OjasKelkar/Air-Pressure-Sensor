from setuptools import find_packages, setup
from typing import List

def get_requirements()->list[str]:
    requirements_list = list[str]=[]
    return requirements_list

setup(
name = "APS",
version = "0.0.1",
author="Ojas",
author_email="kelkarork0308@gmail.com",
packages= find_packages(),
install_requires = ["pymongo"]

)