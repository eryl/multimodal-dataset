# coding: utf-8

from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='multimodal',
      version='0.1',
      description='Multimodal datasets',
      url='http://ghetto.sics.se/ylipaa/multimodal',
      author='Erik Ylipää',
      author_email='erik.ylipaa@ri.se',
      license='MIT',
      packages=['multimodal'],
      install_requires=[],
      dependency_links=[],
      zip_safe=False)
