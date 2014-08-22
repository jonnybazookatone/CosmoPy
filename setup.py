from __future__ import print_function
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand
import io
import codecs
import os
import sys

import CosmoPhotoz as photoz

here = os.path.abspath(os.path.dirname(__file__))

def read(*filenames, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)

#long_description = read('README.md', 'CHANGES.md')

# Files required by the package
cosph_files =  ['data/2slaq_small.csv']
root_files = ['README.md', 'CHANGES.md']

#class PyTest(TestCommand):
#    def finalize_options(self):
#        TestCommand.finalize_options(self)
#        self.test_args = []
#        self.test_suite = True
#
#    def run_tests(self):
#        import pytest
#        errcode = pytest.main(self.test_args)
#        sys.exit(errcode)

setup(
    name='CosmoPhotoz',
    version=photoz.__version__,
    url='http://github.com/COINtoolbox/COSMOPhotoz/CosmoPy',
    license='GNU Public License',
    author=photoz.__author__,
    #tests_require=['pytest'],
    #cmdclass={'test': PyTest},
    #test_suite='sandman.test.test_sandman',
    #extras_require={
    #    'testing': ['pytest'],
    #}
    install_requires=['matplotlib>=1.3.1',
                      'numpy>=1.8.2',
                      'pandas>=0.14.1',
                      'patsy>=0.3.0',
                      'scikit-learn>=0.15.1',
                      'scipy>=0.14.0',
                      'seaborn>=0.3.1',
                      'statsmodels>=0.5.0'],
    author_email=photoz.__email__,
    description=photoz.__doc__,
    long_description=photoz.__doc__,
    packages=['CosmoPhotoz'],
    package_data={
                 'CosmoPhotoz': cosph_files,
                 '': root_files
                 },
    scripts=['CosmoPhotoz/run_glm.py'],
    include_package_data=True,
    platforms='any',
    classifiers = [
        'Programming Language :: Python',
        'Development Status :: 3 - Alpha',
        'Natural Language :: English',
        'Environment :: X11 Applications',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Operating System :: OS Independent',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Astronomy',
        ],
)