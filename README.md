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


