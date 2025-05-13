import setuptools

setuptools.setup(
    name='muse',
    version='0.1.0',
    author='Kenton',
    author_email='kwokkenton@gmail.com',
    description='Short description',
    long_description=open('readme.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/you/my_project',
    license='MIT',
    packages=setuptools.find_packages(),
    # package_dir={"": "captioner"},
    install_requires=[
        'torch',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
    python_requires='>=3.7',
)
