# @robotodio

--------------------------

## 1. Introduction

## 2. Repo Structure

Inside each directory/file wou will find:

- **notebooks:** Jupyter Notebooks
- **scripts:** Code to execute from terminal and interact with the frontend
- **words_list:** Collections of words used in the code
- **requirements_pip.txt:** Required Python libraries in pip format
- **requirements_conda.txt:** Required Python libraries in conda format
- **twitter_api_keys.json:** File with Twitter tokens

Directory structure:

```
project_robotodio
    |
    ├ notebooks
    |   ├ 01-expanded-model.ipynb
    |   ├ 02-multilingual-model.ipynb
    |   └ 03-performance-test.ipynb
    |
    ├ scripts
    |   └ robotodio.py
    |
    ├ words_list
    |   ├ racist_words.txt
    |   └ spanish_stopwords.txt
    |
    ├ requirements_pip.txt
    ├ requirements_conda.txt
    └ twitter_api_keys.json
```

## 3. Requirements and Data Adquisition

### 3.1. Install Requirements

Using pip:
```
pip3 install -r requirements_pip.txt
```

Using conda:
```
conda create --name [env_name] --file requirements_conda.txt
```

Activate enviroment:
```
conda activate [env_name]
```

## 4. Execute code from terminal

Enter the following command into the directory where the .py file is located.
```
streamlit run robotodio.py
```

## 5. Bibliography

- http://docs.tweepy.org/en/latest/
- https://github.com/unitaryai/detoxify
- http://rios.tecnm.mx/cdistribuido/recursos/MinDatScr/MineriaScribble.html
