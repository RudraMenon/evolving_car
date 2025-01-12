from setuptools import setup, find_packages

install_requires = [
    'matplotlib',
    'numpy',
    'opencv-python'
]

torch_requirements = [
    'torch',
    'torchvision',
    'torchaudio'
]

dependency_links = [
    'https://download.pytorch.org/whl/cu118'
]

setup(
    name='genetic_car',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=install_requires,
    extras_require={
        'torch': torch_requirements
    },
    dependency_links=dependency_links,
    author='Rudra Menon',
    description='A genetic algorithm for training a self-driving car',
    python_requires='>=3.9',
)
