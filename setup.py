import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires=["hyperspy"]

try:
    import PyQt5  # noqa
except ImportError:
    try:
        import PySide2  # noqa
    except ImportError:
        install_requires.append('PyQt5==5.14.0')

setuptools.setup(
    name="epsic_tools",
    package_dir={'epsic_tools':'epsic_tools'},
    version="0.0.1",
    author="ePSIC",
    description="A set of tools to analyse ePSIC data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=install_requires,   
    include_package_data=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
)
