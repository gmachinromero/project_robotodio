# @robotodio

--------------------------

## 1. Introduction

## 2. Repo Structure

- **notebooks:** Jupyter Notebooks
- **scripts:** Code to execute from terminal
- **requirements_pip.txt:** Required Python libraries in pip format
- **requirements_conda.txt:** Required Python libraries in conda format
- **spanish_stopwords.txt:** Spanish stop words collection
- **twitter_api_keys.json:** File with Twitter tokens

## 3. Requirements and Data Adquisition

### 3.1. Install Requirements

Using pip:

```
pip3 install -r requirements_pip.txt
```
Using conda:

```
conda create --name <env> --file requirements_conda.txt
```

## 4. Execute code from terminal

```
streamlit run robotodio.py
```
