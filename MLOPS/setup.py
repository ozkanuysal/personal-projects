import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()
    
__version__ = "0.0.1"
REPO_NAME = "personal_projects"
SRC_REPO = "mlops"


setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    description="Personal projects",
    Long_description = long_description,
    long_description_content = "text/makdown",
    package_dir = {"":"src"},
    packages = setuptools.find_packages(where="src")
)
