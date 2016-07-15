try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
        'description': 'Neural Network for calculating risk',
        'author': 'Charis - Nicolas Georgiou',
        'url': 'Url-download',
        'download_url': 'download url put',
        'author_email': 'reloxz@gmail.com',
        'version': '0.1',
        'install_requires': [],
        'packages': ['neuralRisk'],
        'scripts': [],
        'name': 'Neural_Risk_Assesment'
}

setup(**config)
