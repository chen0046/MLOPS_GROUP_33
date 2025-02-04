# (Draft name) Graph Attention Network (GAT) on the CORA dataset
## MLOPS project description

This repository showcases the project work completed by Group 33 as part of the course [02476 Machine Learning Operations](https://kurser.dtu.dk/course/02476) offered at DTU. The members of Group 33 are: 

Chenxi Cai - s205420\
Qiwen Huang - s242640\
Edwin R. Ranjitkumar - s140035\
Alex Richard Hansen - s147082

### Overall goal of the project
The goal of this project is to explore the use of graph neural networks with attention mechanisms to analyze citation relationships between papers, aiming to classify the academic topics of the papers.

### What framework are you going to use 
Since the problem can be categorized as having a graph structure, we plan to use an extension to the PyTorch framework called PyTorch Geometric (PyG). 

### How do you intend to include the framework into your project?
We plan on utilizing PyTorch Geometric (PyG) to implement our model, as it provides efficient tools and utilities for the development of graph based machine learning models. 

### What data are you going to run on (initially, may change)

We are using the publicly available Cora dataset, which is a widely used benchmark dataset in machine learning and graph-based research. It consists of 2,708 scientific papers classified among seven classes (topics).

### What models do you expect to use

We intend to implement a Graph Attention Networks (GATs) using Pytorch Geometric library and train the model on the Cora dataset.

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
