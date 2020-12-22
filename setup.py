import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="double-hard-debias",  # Replace with your own username
    version="0.1",
    author="Haswanth Aekula, Sugam Garg, Animesh Agarwal",
    author_email="haswanth.kumar.39@gmail.com",
    description="Double-Hard Debias",
    long_description=long_description,
    long_description_content_type="text/markdown",
    download_url="https://github.com/hassiahk/Double-Hard-Debias",
    packages=setuptools.find_packages(exclude=['notebooks', 'data']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
