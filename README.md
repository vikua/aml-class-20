# Automated Machine Learning

This repository holds the slides and examples of a class on Automated Machine Learning
as Jupyter notebooks.

# Requirements
- Python 3.7 and pip (see guides on installation for your operating system)
- [Jupyter](https://jupyter.org/install)
- Optional, but recommended: [virtualenv](https://virtualenv.pypa.io/en/latest/installation.html)
- Optional, but recommended: [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/install.html)
- Optional: [Graphviz](https://graphviz.org/download/)

# Setup

### Clone repository
```bash
git clone https://github.com/vikua/aml-class-20.git
cd aml-class-20
```

### Create new virtual environment and install dependencies

```bash
mkvirtualenv aml
```

Optiopnal step, env should be activated by default once created:
```bash
workon aml
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Create ipython kernel:
```bash
python -m ipykernel install --user --name aml
```

### Start Jupyter Notebook

1. At the command line, run `jupyter notebook`
2. Open your web browser to the directed url
3. Open ipynb file of interest


# Table of content

## 1 Introduction

### [1.1 What is Automated Machine Learning?](https://github.com/vikua/aml-class-20/blob/master/part_1/1.1%20What%20is%20AML%3F.pdf)
### [1.2 Machine Learning 101](https://github.com/vikua/aml-class-20/blob/master/part_1/1.2%20Machine%20Learning%20101.ipynb)
### [1.3 Model Selection and Assessment](https://github.com/vikua/aml-class-20/blob/master/part_1/1.3%20Model%20Selection%20and%20Assessmen.pdf)
### [1.4 Hyper-parameter Tuning](https://github.com/vikua/aml-class-20/blob/master/part_1/1.4%20Hyper-parameter%20Tuning.ipynb)
### [1.5 Debugging and Improving ML Models](https://github.com/vikua/aml-class-20/blob/master/part_1/1.5%20Debugging%20and%20Improving%20ML%20Models.ipynb)

## II Tools & Techniques for Automated Machine Learning

### [2.1 Machine Learning Pipelines](https://github.com/vikua/aml-class-20/blob/master/part_2/2.1%20Machine%20Learning%20Pipelines.ipynb)
### [2.2 Bayesian Hyper-parameter Optimization](https://github.com/vikua/aml-class-20/blob/master/part_2/2.2%20Bayesian%20Hyper-parameter%20Optimization.ipynb)
### [2.3 Pipeline Optimization](https://github.com/vikua/aml-class-20/blob/master/part_2/2.3%20Pipeline%20Optimization.ipynb)
### [2.4 Advanced Topics](https://github.com/vikua/aml-class-20/blob/master/part_2/2.4%20Advanced%20Topics%20in%20ML.pdf)

## III Create Value using Automation

### [3.1 Democratizing Machine Learning](https://github.com/vikua/aml-class-20/blob/master/part_3/3%20Creating%20value%20with%20AML.pdf)
### [3.2 Model Factories](https://github.com/vikua/aml-class-20/blob/master/part_3/3%20Creating%20value%20with%20AML.pdf)
### [3.3 Continous Learning](https://github.com/vikua/aml-class-20/blob/master/part_3/3%20Creating%20value%20with%20AML.pdf)
