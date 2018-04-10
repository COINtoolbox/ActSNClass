"""Configure actsnclass instalation."""

from setuptools import setup, find_packages
import actsnclass

def readme():
    """Return README.md file."""
    with open('README.md') as my_doc:
        return my_doc.read()

setup(name='actsnclass',
      version = actsnclass.__version__,
      description = 'Python SN photometric classifier using Active Learning',
      long_description = readme(),
      url = 'https://github.com/COINtoolbox/ActSNClass',
      author = actsnclass.__author__,
      author_email = actsnclass.__email__,
      license = 'GPL3',
      packages = find_packages(),
      install_requires = ['numpy=1.8.2',
                          'matplotlib=1.3.1',
                          'pandas=0.19.2',
                          'scipy=0.18.1',
                          'astropy=2.0.1',
                          'libact=0.1.3'
      ],
      include_package_data=True,
      scripts=['examples/runactsnclass.py'],
      package_dir={'actsnclass': 'actsnclass',
                   'data_functions':'actsnclass/data_functions',
                   'data':'actsnclass/data', 'actConfig':'actsnclass/actConfig',
                   'analysis_functions':'actsnclass/analysis_functions',
                   'lc_functions':'actsnclass/lc_functions'},
      package_data = {'actsnclass/data':'cross_Validation_labels.txt',
                      'actsnclass/data':'crossValidation_lightcurves.txt',
                      'actsnclass/data':'train_labels.txt', 
                      'actsnclass/data':'train_lightcurves.txt'},
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
