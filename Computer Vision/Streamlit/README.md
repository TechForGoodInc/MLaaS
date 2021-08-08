# Reproducing this web app
To recreate this web app on your own computer, do the following.

### Create conda environment
Firstly, we will create a conda environment called *multipage*
```
conda create -n multipage python=3.7.9
```
Secondly, we will login to the *multipage* environement
```
conda activate multipage
```
### Install prerequisite libraries

Download requirements.txt file

```
wget https://raw.githubusercontent.com/TechForGoodInc/MLaaS/main/Computer%20Vision/Streamlit/requirements.txt

```

Pip install libraries
```
pip install -r requirements.txt
```

###  Launch the app

```
streamlit run app.py
```
