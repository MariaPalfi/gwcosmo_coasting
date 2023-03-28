import sys
from setuptools import setup

setup_requires = ['setuptools >= 30.3.0']
if {'pytest', 'test', 'ptr'}.intersection(sys.argv):
    setup_requires.append('pytest-runner')
if {'build_sphinx'}.intersection(sys.argv):
    setup_requires.extend(['recommonmark',
                           'sphinx'])

def readme():
    with open('README.md') as f:
        return f.read()

with open('requirements.txt') as f:
    reqs = f.read()

setup(name='gwcosmo_coasting',
      version='1.0.0',
      description='A package to estimate cosmological parameters using gravitational-wave observations',
      #url='https://git.ligo.org/lscsoft/gwcosmo',
      author='Cosmology R&D Group',
      author_email='cbc+cosmo@ligo.org',
      license='GNU',
      packages=['gwcosmo_coasting', 'gwcosmo_coasting.likelihood', 'gwcosmo_coasting.prior', 'gwcosmo_coasting.utilities','gwcosmo_coasting.plotting'],
      package_dir={'gwcosmo': 'gwcosmo'},
      scripts=['bin/gwcosmo_coasting_single_posterior', 'bin/gwcosmo_coasting_combined_posterior', 'bin/gwcosmo_coasting_compute_pdet', 'bin/gwcosmo_coasting_pixel_dag'],
      include_package_data=True,
      install_requires=reqs,
      setup_requires=setup_requires,
      zip_safe=False)

