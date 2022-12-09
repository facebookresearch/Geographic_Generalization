from setuptools import setup, find_packages

# Read README and requirements
with open("README.md", encoding="utf8") as f:
    readme = f.read()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="interplay_project",
    packages=find_packages(exclude=["tests"]),
    long_description=readme,
    install_requires=requirements,
)
