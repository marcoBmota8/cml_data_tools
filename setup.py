"""
Data tools developed by the Computational Medicine Lab at VUMC
"""
from os.path import abspath, dirname, join
from setuptools import setup

readme_path = join(abspath(dirname(__file__)), 'README.md')
with open(readme_path, encoding='utf-8') as file:
    DESCRIPTION = file.read()

setup(
    name='cml-data-tools',
    description=__doc__.strip(),
    url='https://github.com/marcoBmota8/cml_data_tools/tree/master',
    py_modules=['cml_data_tools'],
    license='BSD',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Information Analysis',
    ],
    author='John M. Still, Thomas A. Lasko',
    maintainer='John M. Still',
    long_description=DESCRIPTION,
    long_description_content_type='text/markdown',
    # Install data
    zip_safe=False,
    include_package_data=True,
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'fast-intensity>=0.4',
    ],
)
