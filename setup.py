from setuptools import setup, find_packages

setup(
    name="dronesim2sim",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.24.0,<2.0.0",
        "torch>=2.0.0",
        "jax>=0.4.13",
        "jaxlib>=0.4.13",
        "gymnasium>=1.0.0",
        "pybullet>=3.2.0",
        "matplotlib>=3.0.0",
        "pandas>=1.5.0,<2.0.0",
        "protobuf>=3.20.0,<4.0.0"
    ],
    python_requires=">=3.8",
    description="A framework for drone simulation and control",
    author="Shengyang Wang",
    author_email="shengyang.wang2004@gmail.com",
    url="https://github.com/Wangshengyang2004/DroneSim2Sim",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
) 