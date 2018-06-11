from setuptools import setup, find_packages

setup(name='foodmenurecognition',
      version='0.1.0',
      description='Food Menu Recognition',
      url='https://wwww.marcvaldivia.com',
      author='Marc Valdivia',
      packages=find_packages(),
      zip_safe=False, install_requires=['pandas', 'scikit-learn', 'numpy'])
