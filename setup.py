from setuptools import setup, find_packages

setup(
    name='ptopofl',
    version='1.0.0',
    description='Privacy-Preserving Personalised Federated Learning via Persistent Homology',
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=[
        'numpy>=1.24',
        'scipy>=1.10',
        'scikit-learn>=1.3',
        'matplotlib>=3.7',
        'tqdm>=4.65',
    ],
)
