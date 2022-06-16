from setuptools import setup, find_packages

with open("requirements.txt") as f:
    reqs = []
    for line in f:
        line = line.strip()
        # let's also ignore empty lines and comments
        if not line or line.startswith("#"):
            continue
        if "https://" in line:
            tail = line.rsplit("/", 1)[1]
            tail = tail.split("#")[0]
            line = tail.replace("@", "==").replace(".git", "")
        reqs.append(line)

setup(
    name="observational_fear",
    version="0.0.1",
    author="Ruairi O'Sullivan",
    author_email="ruairi.osullivan.work@gmail.ie",
    packages=find_packages(),
    python_requires=">=3.5",
    install_requires=reqs,
)
