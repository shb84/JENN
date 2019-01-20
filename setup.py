import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="genn",
    version="0.1.0",
    author="Steven H. Berguin",
    author_email="stevenberguin@gmail.com",
    description="Gradient Enhanced Neural Network (genn)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shb84/genn.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    package_data={'demo': ['../demo/demo_notebook.ipynb', '../demo/demo_test_data.csv', '../demo/demo_train_data.csv'],
                  'docs': ['../docs/theory.pdf']},
)
