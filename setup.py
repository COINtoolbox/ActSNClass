"""Configure snactclass instalation."""

from setuptools import setup, find_packages
import snactclass

def readme():
    """Return README.md file."""
    with open('README.md') as my_doc:
        return my_doc.read()

setup(name='snactclass',
      version = snactclass.__version__,
      description = 'Python SN photometric classifier using Active Learning',
      long_description = readme(),
      url = 'https://github.com/emilleishida/snactclass',
      author = snactclass.__author__,
      author_email = snactclass.__email__,
      license = 'GPL3',
      packages = find_packages(),
      install_requires = ['numpy>=1.8.2',
                          'matplotlib>=1.3.1',
                          'pandas>=0.19.2',
                          'scipy>=0.18.1',
                          'astropy>=2.0.1'
      ],
      include_package_data=True,
      scripts=['examples/runSNActClass.py'],
      package_dir={'snactclass': 'snactclass',
                   'data_functions':'snactclass/data_functions',
                   'data':'snactclass/data', 'actConfig':'snactclass/actConfig',
                   'analysis_functions':'snactclass/analysis_functions',
                   'lc_functions':'snactclass/lc_functions'},
      package_data = {'snactclass/data':'cross_Validation_labels.txt',
                      'snactclass/data':'crossValidation_lightcurves.txt',
                      'snactclass/data':'train_labels.txt', 
                      'snactclass/data':'train_lightcurves.txt'},
      zip_safe=False,
      classifiers=[
        'Programming Language :: Python',
        'Natural Language :: English',
        'Environment :: X11 Applications',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Astronomy',
        ])
