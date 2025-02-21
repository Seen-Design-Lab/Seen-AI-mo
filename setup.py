from setuptools import setup, find_packages

setup(
    name='seenai',
    version='0.1.0',
    description='AI-powered UX research tool',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/your-username/seenai',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.19.2',
        'pandas>=1.1.3',
        'tensorflow>=2.3.1',
        'scikit-learn>=0.23.2',
        'matplotlib>=3.3.2',
        'seaborn>=0.11.0'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)
