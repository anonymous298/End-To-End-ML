from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path: str) -> List[str]:
    '''
    This function returns the list of Requirements
    '''

    with open(file_path) as file:
        lines = file.readlines()
        lines = [word.replace('\n','') for word in lines if word != '-e .']

    return lines

setup(
    name='End-To-End-ML',
    version='0.0.1',
    author='Talha',
    author_email='tackletalha@gmail.com',
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt')
)