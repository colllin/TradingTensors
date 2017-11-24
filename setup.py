from setuptools import setup


setup(
    name='tradingtensor',
    version='0.1',
    author='henryBee',
    install_requires=[
        'numpy',
        'pandas',
        'tensorflow'
    ],
    extras_require = {
        'swisseph':  [
            "swissseph",
            'talib']
    })