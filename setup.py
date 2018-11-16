import os

from setuptools import setup

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def read(fname):
    return open(os.path.join(ROOT_DIR, fname)).read()


with open('requirements.txt') as reqs:
    requirements = [line.strip().split("==")[0] for line in reqs.readlines()]

setup(
    name="estee",
    version="0.2",
    author="Estee team",
    author_email="stanislav.bohm@vsb.cz",
    description="Experimental Scheduler Training EnvironmEnt",
    license="MIT",
    keywords="scheduling, simulation",
    url="http://estee.readthedocs.io",
    packages=["schedsim"],
    install_requires=requirements,
    long_description=read('README.md'),
)
