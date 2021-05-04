# @robotodio

--------------------------

## 1. Introduction

It is a fact that nowadays, social networks are part of our day to day, they allow us to communicate with friends, family, or even celebrities or people who are not so close. These promote the rapprochement and the relationship between people, although like everything else, they can also cause discussions and trigger attacks and insults among the users of these social networks.

This is a serious problem that companies have to face, where is the limit?, who sets it? how is it monitored?

It has been shown that with the amount of messages that are produced daily on social networks, it is practically impossible to manually check which ones comply with the community norms and which ones do not. Additionally, there are studies that show that those people whose job it is to monitor and review aggressive or hate messages on these platforms, are more likely to suffer from depression or stressful situations.

The key here is to develop systems and algorithms that allow us to accurately identify these hateful messages, or to do an initial screening and leave those more ambiguous to the review of the human eye, facilitating the work of people, and the user experience of the community within these social networks.

With this objective, the robotodio project was born, to identify hate speech on Twitter, to improve social networks and make the population aware of the polarity of their messages.

## 2. Repo Structure

Inside each directory/file wou will find:

- **notebooks:** Jupyter Notebooks with explanations
- **scripts:** Code to execute from terminal and interact with the frontend
- **words_list:** Collections of words used in the code
- **requirements_pip.txt:** Required Python libraries in pip format
- **requirements_conda.txt:** Required Python libraries in conda format
- **template_twitter_api_keys.json:** File with Twitter tokens (you have to request your own tokens to Twitter)

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

As you can see, the requirements files contain a lot of libraries associated with anaconda notebooks. If you want to run the code exclusively in the terminal, you will only need the libraries defined in the requirements_terminal.

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

Run the following command into the directory where the .py file is located ([~/.../.../etc]/project_robotodio/scripts).
```
cd ~/.../.../project_robotodio/scripts
streamlit run robotodio.py
```

## 5. Bibliography

- http://docs.tweepy.org/en/latest/
- https://github.com/unitaryai/detoxify
- http://rios.tecnm.mx/cdistribuido/recursos/MinDatScr/MineriaScribble.html
