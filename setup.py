
from Cython.Build.Dependencies import cythonize
import numpy
from setuptools import setup, find_packages
from setuptools.extension import Extension


# This is a list of files to install, and where
# (relative to the 'root' dir, where setup.py is)
# You could be more specific.
# Build the dist with:
#    python setup.py sdist
# More ifo on :https://www.digitalocean.com/community/tutorials/how-to-package-and-distribute-python-applications
morphLibExtension = Extension( "pyVET.morphing", 
                               sources = ['pyVET/morphing.pyx'],
                               include_dirs=[numpy.get_include()],
                               extra_compile_args = ['-fopenmp'],
                               extra_link_args = ['-fopenmp'] )


setup( name = "pyVET",
    version = "1.0.0",
    description = "pyVET",
    author = "Andres Perez Hortal",
    author_email = "andresperezcba@gmail.com",
    # Name the folder where your packages live:
    # (If you have other packages (dirs) or modules (py files) then
    # put them into the package directory - they will be found 
    # recursively.)
    
    packages = find_packages(),
    # package_data={'pyWAT': ['icons/*']},
    ext_modules =cythonize( [ morphLibExtension]),
    include_package_data = True,
    # packages = ['pyWrf'],
    # 'package' package must contain files (see list above)
    # I called the package 'package' thus cleverly confusing the whole issue...
    # This dict maps the package name =to=> directories
    # It says, package *needs* these files.
    # package_data = {'wrfUtilities' : ["README"] , 'wrfUtilities' : ["parameters.ini"] },
    # 'runner' is in the root.
    # scripts = ["runner"],
    long_description = """
    Variational Echo Tracking
    """ 
    
 ) 
