#To build our application as pacakge
from setuptools import find_packages,setup
from typing import List

def get_requirements(file_path:str)->List[str]:
    HYPHEN_E_DOT="-e ."
    '''
    This function will return list of requirements
    '''
    requirement =[]
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
setup(
    name='mlproject',
    version='0.0.1',
    author='Pramopras',
    author_email='pramopras@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
