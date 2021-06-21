
# ------------------------------------------------------------------------------
# Python libraries
# ------------------------------------------------------------------------------
# Operating system
import os

# Reading files with different formats
import json

# Data wrangling
import re
import pandas as pd
import numpy as np

# Data Visualitation
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import streamlit as st

# Twitter API
import tweepy

# Hate speech detection
from detoxify import Detoxify



# ------------------------------------------------------------------------------
# API Twitter credentials and loggin
# ------------------------------------------------------------------------------

# Create a handler instance with key and secret consumer, and pass the tokens
auth = tweepy.OAuthHandler(
    os.environ["consumer_key"],
    os.environ["consumer_secret"])

auth.set_access_token(
    os.environ["access_token"],
    os.environ["access_token_secret"])
    
# Construct the API instance
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

# Check credentials
if(api.verify_credentials):
    print('-'*30)
    print("Logged In Successfully.")
    print('-'*30)
else:
    print("Error -- Could not log in with your credentials.")



# ------------------------------------------------------------------------------
# Streamlit appereance
# ------------------------------------------------------------------------------

# Appeareance
st.set_page_config(
    page_title = "robotodio",
    page_icon = "👁",
    layout = "wide",
    initial_sidebar_state = "expanded"
    )

# Title
with st.beta_container():
    st.write("""
    # Robotodio 👁 🔪
    *Analizando la agresividad en Twitter*
    """)
    st.markdown("---")

# Sidebar
st.sidebar.title('Introduce una cuenta de Twitter')
target = st.sidebar.text_input("sin @")

# Reloj
with st.spinner('Tened paciencia y tendréis ciencia...⏳'):


    # ------------------------------------------------------------------------------
    # Twitter extractor and processor
    # ------------------------------------------------------------------------------

    # ----------------------- Tweet downloader -------------------------------------

    def tweets_iterator(target, n_items):
        '''
        Returns an iterator of tweets.

            Parameters:
                target (str): The user name of the Twitter account.
                n_items (int): Number of tweets downloaded.

            Returns:
                tweets (ItemIterator): an iterator of tweets.
        '''
        # Instantiate the iterator
        tweets = tweepy.Cursor(
            api.user_timeline,
            screen_name=target,
            include_rts=False,
            exclude_replies=False,
            tweet_mode='extended').items(n_items)
        
        # Returns iterator
        return tweets

    # Tweets list (iterator)
    tweets = tweets_iterator(target, n_items=20)

    # Read through the iterator, and export the info to a Pandas DataFrame
    all_columns = [np.array([
        tweet.full_text,
        tweet.user.screen_name,
        tweet.id,
        tweet.user.followers_count,
        tweet.source,
        tweet.created_at,
        tweet.lang,
        len(tweet.full_text),
        tweet.favorite_count,
        tweet.retweet_count,
        re.findall(r"#(\w+)", tweet.full_text)
    ]) for tweet in tweets]

    # Export the list of tweets to a dataframe
    df = pd.DataFrame(
        data=all_columns,
        columns=['tweet', 'account', 'id', 'followers', 'source', 'date',
                'language', 'length', 'likes', 'RTs', 'hashtags']
                )


    # ----------------------- Account data extractor -------------------------------

    tweets = tweets_iterator(target, n_items=1)

    # Read through the iterator, and export the info to a Pandas DataFrame
    data_account = [
        np.array([
            tweet.user.screen_name,
            tweet.user.name,
            tweet.user.description,
            tweet.user.created_at,
            tweet.user.friends_count,
            tweet.user.followers_count,
            tweet.user.statuses_count
            ]) for tweet in tweets
            ]

    # Export the list of features to a dataframe
    df_account = pd.DataFrame(
        data=data_account,
        columns=['account', 'account_name', 'bio_description', 'creation_date',
                'friends', 'followers', 'tweets']
                )

    df_account = pd.melt(df_account)


    # ----------------------- Data cleansing ---------------------------------------

    # Characters to remove
    spec_chars = ['\n', '\t', '\r']
    # Replace defined characters with a whitespace
    for char in spec_chars:
        df['tweet'] = df['tweet'].str.strip().replace(char, ' ')
    # Split and re-join each tweet
    df['tweet'] = df['tweet'].str.split().str.join(" ")


    # ----------------------- Hate speech level prediction -------------------------

    # Instance the model
    results = Detoxify('multilingual').predict(list(df['tweet']))

    # Add the new info to the previous DataFrame
    df['toxicity'] = results['toxicity']

    # Define a class for each tweet (toxic or non-toxic)
    df['class'] = df['toxicity'].apply(
        lambda toxicity: 'toxic' if toxicity >= 0.5 else 'non-toxic'
        )
        
    racist_words = ['africana', 'africano', 'china', 'chino', 'extranjera', 'extranjero',
                    'gitana', 'gitano', 'india', 'indigena', 'indio', 'inmigrante',
                    'latina', 'latino', 'mantera', 'mantero', 'mena', 'mora', 'moro',
                    'negra', 'negrata', 'negro', 'paki', 'panchita', 'panchito', 'sudaca',
                    'tiraflecha']

    df['racist'] = df['tweet'].apply(
        lambda tweet: 'possibly racist' if any (
            word in tweet for word in racist_words
            ) else 'non-racist')

    # Calculate average toxicity level
    scoring_average = {'variable': ['avg_toxicity', 'racist_score'],
                    'value': [df['toxicity'].mean(), len(df[df['racist']=='possibly racist'])/len(df['racist'])]}

    df_average = pd.DataFrame(scoring_average)

    # Principal target account info
    df_account = pd.concat([df_account, df_average], ignore_index=True)

    # Top 5 toxic tweets
    df_top_5 = df[['account', 'date', 'tweet', 'likes', 'RTs', 'toxicity', 'class']]\
            .sort_values('toxicity', ascending=False).head()



    # -----------------------------------------------------------------------------
    # Streamlit output text
    # -----------------------------------------------------------------------------

    # Principal target account info
    cuenta_twitter = df_account['value'][0]
    md_cuenta_twitter = f"**@Cuenta:** *{cuenta_twitter}*"

    nombre_cuenta = df_account['value'][1]
    md_nombre_cuenta = f"**Nombre:** *{nombre_cuenta}*"

    descripcion = df_account['value'][2]
    md_descripcion = f"**Bio:** *{descripcion}*"

    fecha_inicio = df_account['value'][3]
    md_fecha_inicio = f"**Creación:** *{fecha_inicio}*"

    seguidos = df_account['value'][4]
    md_seguidos = f"**Sigue:** *{seguidos}*"

    seguidores = df_account['value'][5]
    md_seguidores = f"**Le siguen:** *{seguidores}*"

    num_tweets = df_account['value'][6]
    md_num_tweets = f"**Tweets:** *{num_tweets}*"

    toxicidad_media = round(df_account['value'][7]*100, 2)
    md_toxicidad_media = f"**Toxicidad:** *{toxicidad_media}%*"

    racismo = round(df_account['value'][8]*100, 2)
    md_racismo = f"**Racismo:** *{racismo}%*"

    with st.beta_container():
        
        col1, col2 = st.beta_columns(2)

        with col1:    
            st.markdown(md_cuenta_twitter)
            st.markdown(md_nombre_cuenta)
            st.markdown(md_descripcion)
            st.markdown(md_fecha_inicio)
            st.markdown(md_seguidos)
            st.markdown(md_seguidores)
            st.markdown(md_num_tweets)
            st.markdown(md_toxicidad_media)
            st.markdown(md_racismo)

    st.markdown("---")

    with st.beta_container():
        st.subheader('Top 5 tweets más agresivos')
        st.table(data=df_top_5)

    st.markdown("---")



    # -----------------------------------------------------------------------------
    # Streamlit Graphics
    # -----------------------------------------------------------------------------

    # ----------------------- Toxicity vs. Likes & RTs ----------------------------
    # Figure and axes
    fig_1, (ax1, ax2) = plt.subplots(1, 2, figsize=(18,5), sharex=True, sharey=True)

    # Titles
    ax1.set_title("Likes vs. Toxicity", y=1.02, fontsize=10, fontweight='bold')
    ax2.set_title("RTs vs. Toxicity", y=1.02, fontsize=10, fontweight='bold')

    # Axes content
    sns.scatterplot(ax=ax1, data=df, x='toxicity', y='likes', hue='class', alpha=0.6, legend=False)
    sns.scatterplot(ax=ax2, data=df, x='toxicity', y='RTs', hue='class', alpha=0.6, legend=False);

    st.pyplot(fig_1)

    st.markdown("---")


    # ----------------------- Toxicity histogram ----------------------------------
    # Figure and axes
    fig_2, ax = plt.subplots(figsize=(18,5))

    # Titles
    ax.set_title("Toxicity histogram", y=1.02, fontsize=10, fontweight='bold')

    # Toxicity evolution through time
    sns.histplot(ax=ax, data=df, x='toxicity', hue='class', legend=True, fill=True);

    st.pyplot(fig_2)

    st.markdown("---")


    # ----------------------- Evolution of toxicity through time ------------------
    # Figure and axes
    fig_3, ax = plt.subplots(1, figsize=(18,5))

    # Titles
    ax.set_title("Toxicity time serie", y=1.02, fontsize=10, fontweight='bold')

    # Toxicity evolution through time
    sns.lineplot(ax=ax, data=df, x="date", y="toxicity");

    st.pyplot(fig_3)

    st.markdown("---")


    # ----------------------- WordCloud --------------------------------------------
    # Combine all tweets
    text = " ".join(review for review in df['tweet'])

    spanish_stopwords = ['de', 'la', 'que', 'el', 'en', 'y', 'a', 'los', 'del', 'se', 'las', 'por', 'un', 'para', 'con', 'no', 'una', 'su', 'al', 'lo', 'como', 'más', 'pero', 'sus', 'le', 'ya', 'o', 'este', 'sí', 'porque', 'esta', 'entre', 'cuando', 'muy', 'sin', 'sobre', 'también', 'me', 'hasta', 'hay', 'donde', 'quien', 'desde', 'todo', 'nos', 'durante', 'todos', 'uno', 'les', 'ni', 'contra', 'otros', 'ese', 'eso', 'ante', 'ellos', 'e', 'esto', 'mí', 'antes', 'algunos', 'qué', 'unos', 'yo', 'otro', 'otras', 'otra', 'él', 'tanto', 'esa', 'estos', 'mucho', 'quienes', 'nada', 'muchos', 'cual', 'poco', 'ella', 'estar', 'estas', 'algunas', 'algo', 'nosotros', 'mi', 'mis', 'tú', 'te', 'ti', 'tu', 'tus', 'ellas', 'nosotras', 'vosotros', 'vosotras', 'os', 'mío', 'mía', 'míos', 'mías', 'tuyo', 'tuya', 'tuyos', 'tuyas', 'suyo', 'suya', 'suyos', 'suyas', 'nuestro', 'nuestra', 'nuestros', 'nuestras', 'vuestro', 'vuestra', 'vuestros', 'vuestras', 'esos', 'esas', 'estoy', 'estás', 'está', 'estamos', 'estáis', 'están', 'esté', 'estés', 'estemos', 'estéis', 'estén', 'estaré', 'estarás', 'estará', 'estaremos', 'estaréis', 'estarán', 'estaría', 'estarías', 'estaríamos', 'estaríais', 'estarían', 'estaba', 'estabas', 'estábamos', 'estabais', 'estaban', 'estuve', 'estuviste', 'estuvo', 'estuvimos', 'estuvisteis', 'estuvieron', 'estuviera', 'estuvieras', 'estuviéramos', 'estuvierais', 'estuvieran', 'estuviese', 'estuvieses', 'estuviésemos', 'estuvieseis', 'estuviesen', 'estando', 'estado', 'estada', 'estados', 'estadas', 'estad', 'he', 'has', 'ha', 'hemos', 'habéis', 'han', 'haya', 'hayas', 'hayamos', 'hayáis', 'hayan', 'habré', 'habrás', 'habrá', 'habremos', 'habréis', 'habrán', 'habría', 'habrías', 'habríamos', 'habríais', 'habrían', 'había', 'habías', 'habíamos', 'habíais', 'habían', 'hube', 'hubiste', 'hubo', 'hubimos', 'hubisteis', 'hubieron', 'hubiera', 'hubieras', 'hubiéramos', 'hubierais', 'hubieran', 'hubiese', 'hubieses', 'hubiésemos', 'hubieseis', 'hubiesen', 'habiendo', 'habido', 'habida', 'habidos', 'habidas', 'soy', 'eres', 'es', 'somos', 'sois', 'son', 'sea', 'seas', 'seamos', 'seáis', 'sean', 'seré', 'serás', 'será', 'seremos', 'seréis', 'serán', 'sería', 'serías', 'seríamos', 'seríais', 'serían', 'era', 'eras', 'éramos', 'erais', 'eran', 'fui', 'fuiste', 'fue', 'fuimos', 'fuisteis', 'fueron', 'fuera', 'fueras', 'fuéramos', 'fuerais', 'fueran', 'fuese', 'fueses', 'fuésemos', 'fueseis', 'fuesen', 'sintiendo', 'sentido', 'sentida', 'sentidos', 'sentidas', 'siente', 'sentid', 'tengo', 'tienes', 'tiene', 'tenemos', 'tenéis', 'tienen', 'tenga', 'tengas', 'tengamos', 'tengáis', 'tengan', 'tendré', 'tendrás', 'tendrá', 'tendremos', 'tendréis', 'tendrán', 'tendría', 'tendrías', 'tendríamos', 'tendríais', 'tendrían', 'tenía', 'tenías', 'teníamos', 'teníais', 'tenían', 'tuve', 'tuviste', 'tuvo', 'tuvimos', 'tuvisteis', 'tuvieron', 'tuviera', 'tuvieras', 'tuviéramos', 'tuvierais', 'tuvieran', 'tuviese', 'tuvieses', 'tuviésemos', 'tuvieseis', 'tuviesen', 'teniendo', 'tenido', 'tenida', 'tenidos', 'tenidas', 'tened', '', 't', 'si', 'https', 'https', 'co']

    # Create and generate a word cloud image
    wordcloud = WordCloud(
        width=1800,
        height=1000,
        stopwords=spanish_stopwords,
        background_color="white").generate(text)

    # Display the generated image:
    fig_4 = plt.figure(figsize=(8,6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

    with col2:
        st.pyplot(fig_4)

    st.markdown("---")


# Reloj
st.success('Terminado!')