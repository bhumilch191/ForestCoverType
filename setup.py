from setuptools import find_packages
from setuptools import setup
from typing import List


def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as f_obj:
        requirements=f_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]
        return requirements

setup(
    name='Forest-Cover-Type',
    version='0.0.1',
    description="Predict seven different cover types in four different wilderness areas of Forest",
    long_description=open("README.md", encoding="utf-8").read(),
    author='Bhumil',
    author_email='bhumilc88@gmail.com',
    include_dirs=get_requirements('requirements.txt'),
    packages=find_packages(),
    url='https://github.com/bhumilch191/ForestCoverType',
    license='MIT',
    readme="README.md",
    python_requires=">=3.8"
)