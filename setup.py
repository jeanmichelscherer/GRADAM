from setuptools import setup

setup(
  name='gradam',
  version='1.0.0',
  author='Jean-Michel Scherer',
  author_email='jm.scherer@outlook.com',
  packages=['gradam'],
  package_dir={'': 'src'},
  scripts=[],
  url='https://github.com/jeanmichelscherer/GRADAM',
  license='LICENSE.txt',
  description='A FEniCS-based implementation of variationnal phase-field fracture.',
  long_description=open('README.md').read(),
  install_requires=[
      "scipy",
      "matplotlib",
      "numpy",
      "meshio[all]",
#      ["--no-binary=h5py", "h5py"],
  ],
)
