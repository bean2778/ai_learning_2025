[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "ml_formulation"
version = "0.1.0"
description = "Tools for formulating machine learning problems from business requirements"
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
readme = "README.md"
requires-python = ">= 3.8"
dependencies = [
    "numpy >= 1.21.0",
    "pandas >= 1.3.0",
    "scikit-learn >= 1.0.0",
    "matplotlib >= 3.4.0",
    "seaborn >= 0.11.0"
]

[project.optional-dependencies]
dev = [
    "pytest >= 7.0.0",
    "jupyter >= 1.0.0",
    "ipykernel >= 6.0.0"
]

[project.license]
text = "GPL-3.0-or-later"