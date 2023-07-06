import pandas as pd
import re
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from langdetect import detect
from nltk.tokenize import word_tokenize
from transformers import GPT2Tokenizer, GPT2Config

######################## List of white wine and dessert wine types ########################
white = [
 'White', 'Chardonnay', 'Riesling', 'Rose', 'Sauvignon Blanc',
 'Sparkling', 'Champagne', 'Chenin Blanc', 'Pinot Grigio', 'Côte de Beaune White',
 'Pinot Gris', 'Macônnais White', 'Grüner Veltliner', 'White Blend', 'Sauternes',
 'Prosecco', 'Viognier', 'Albariño', 'Gewürztraminer', 'Cava',
 'Pinot Blanc', 'Côte Chalonnaise White', 'Vinho Verde White', 'Cremant', 'Moscato d\'Asti',
 'Muscadet', 'Rioja White', 'Gavi', 'Verdejo', 'Soave', 'Torrontes',
 'Condrieu', 'Méditerranée', 'Txakoli', 'Müller Thurgau', 'Grenache Rosé',
 'Sémillon', 'Silvaner', 'Saint-Péray', 'Tempranillo Rosé', 'Tokaji Aszú',
 'Furmint', 'Cerasuolo d\'Abruzzo', 'Muscat', 'Pinot Noir Rosé', 'Grenache Blanc',
 'Sauvignon Blanc - Sémillon', 'Sylvaner', 'Vermentino', 'Scheurebe', 'Sekt',
 'Grauburgunder', 'Asti', 'Malagouzia', 'Chasselas', 'Sparkling Shiraz',
 'Pedro Ximenez', 'Rkatsiteli', 'Welschriesling', 'Mtsvane Kakhuri', 'Hárslevelű',
 'Vin Jaune', 'Petite Arvine', 'Savagnin', 'Malvazija Istarska', 'Müller-Thurgau',
 'Airén', 'Auxerrois', 'Verdelho', 'Chardonnay - Torrontés', 'Muscat Bailey A'
]

dessert_wines =[
    'Dessert', 'Sherry', 'Tawny Port', 'Vintage Port', 'Chenin Blanc Dessert',
    'Late Bottled Vintage Port', 'Fortified', 'Vin Santo', 'Single Quinta Vintage Port',
    'White Port', 'Colheita Port', 'Port', 'Ice Wine', 'Riesling Dessert',
    'Crusted Port', 'Gewürztraminer Dessert', 'Pinot Gris Dessert'
]

######################## Data Cleaning: Classification ########################
def extract_varietal(name):
    """
    Given the full name of a wine, extract varietal type if name is in white, red or desset wine lists.
    """
    wine = pd.read_csv("data/wine_raw.csv", index_col=0)
    varietals = list(wine.varietal.value_counts().index)
    varietals.remove("White") 
    varietals.remove("Red")
    for v in varietals:
        match = re.findall("{}".format(v), name)
        if len(match) > 0:
            extracted_varietal = v
            break
        else:
            continue
    try:
        extracted_varietal
    except:
        print("no specific varietal")
        for t in ["Red", "White"]:
            match = re.findall("{}".format(t), name)
            if len(match) > 0:
                extracted_varietal = t
                break
            else:
                continue
    try:
        extracted_varietal
    except:
        extracted_varietal=np.nan
    return extracted_varietal

def classify_wine_type(varietal):
    """
    Classify wine type as white, or nan (for dessert wines and wines with missing varietal)
    """
    if (varietal in remove_wines) | (math.isnan(varietal)):
        wine_type = np.nan
    elif varietal in white:
        wine_type = "white"
    else:
        wine_type = "red"
    return wine_type

def set_wine_type(wine_df):
    """
    This function takes in a wine dataframe and returns a wine dataframe with a 'type' column
    """
    wine_types = []
    for i, row in wine_df.iterrows():
        v = row.varietal
        if pd.isna(v):
            wine_types.append(np.nan)
        elif v in dessert_wines:
            wine_types.append("dessert")
        elif v in white:
            wine_types.append("white")
        else:
            wine_types.append("red")
    wine_df['type'] = wine_types
    return wine_df
    
def detect_language(text):
    """
    Given text, return language origin. This function is utilized to filter out non-english reviews
    """
    lang = detect(text)
    return lang

def remove_foreign_characters(text):
    """
    Remove foreign language and emojis.
    """
    non_ascii_pattern = r'[^\x00-\x7F]'
    cleaned = re.sub(non_ascii_pattern, '', text)
    return cleaned

def remove_numbers(text):
    """
    Remove all numbers from text
    """
    cleaned = re.sub(r"\d+", '', text)
    return cleaned

def clean_reviews_classifier(review):
    """
    Clean reviews to prepare for classification task
    - Remove foreign characters and numbers
    - Remove hyperlinks and special characters
    - Remove stopwords that won't be helpful in classification task
    """
    clean_review = str(review) # ensure all reviews are string datatype
    clean_review = remove_foreign_characters(clean_review) # remove foreign characters
    clean_review = remove_numbers(clean_review) # remove digits
    clean_review = clean_review.lower() # convert to lowercase
    clean_review = re.sub(r"!gif\(giphy.*\)", "", clean_review) # remove gifs
    clean_review = re.sub(r"\(https[^\s]*\)", " ", clean_review) # remove embedded hyperlinks
    clean_review = re.sub(r"https[^\s]*", " ", clean_review) # remove remaining hyperlinks
    clean_review = re.sub(r"amp;", "", clean_review) # remove ampersand
    clean_review = re.sub(r"[\$\*'’\[\]#,%\"^&\.\?\!<>]", "", clean_review) # remove special characters
    clean_review = re.sub(r"[/-_\+\\]", " ", clean_review) # remove more special characters
    
    # split text, remove stopwords then join back
    swords = set(stopwords.words("english"))
    clean_review = clean_review.split()
    clean_review = [w for w in clean_review if w not in swords]
    clean_review = " ".join(clean_review)
    
    return clean_review

#################### Create Different Data Subsets for Different Models ##############################
    
def subset_data_classification():
    """
    Subset data into 6 different categories, based on red or white wine type and three price ranges.
    """
    columns = ['review_id', 'wine_id', 'rating', 'review_text', 'date', 'price', 'type']
    df_r1 = pd.DataFrame(columns=columns)
    df_r2 = pd.DataFrame(columns=columns)
    df_r3 = pd.DataFrame(columns=columns)
    df_w1 = pd.DataFrame(columns=columns)
    df_w2 = pd.DataFrame(columns=columns)
    df_w3 = pd.DataFrame(columns=columns)

    review_count = 2500

    wine_sub = wine[['id', 'price', 'type']]
    wine_sub = wine_sub.groupby('id').max().reset_index()

    while review_count <= 50000:
        reviews = pd.read_csv("data/reviews_{}.csv".format(review_count), index_col = 0)
        reviews = reviews.drop_duplicates() # drop dups for sanity check
        reviews = reviews[reviews['language'] == 'en'] # filter out non-english reviews
        reviews = reviews[['review_id', 'wine_id', 'rating', 'review_text', 'date']]
        # join wine df with reviews, adding price/type columns
        reviews_full = reviews.merge(wine_sub, how='left', left_on='wine_id', right_on='id').drop(columns=['id'])
        # concat with 6 subsector dfs
        r1 = reviews_full[(reviews_full['type'] == 'red') & (reviews_full['price'] <= 15.0)]
        df_r1 = pd.concat([df_r1, r1]).reset_index().drop(columns=['index'])
        r2 = reviews_full[(reviews_full['type'] == 'red') & (reviews_full['price'] > 15.0) & (reviews_full['price'] <= 35.0)]
        df_r2 = pd.concat([df_r2, r2]).reset_index().drop(columns=['index'])
        r3 = reviews_full[(reviews_full['type'] == 'red') & (reviews_full['price'] >= 35.0)]
        df_r3 = pd.concat([df_r3, r3]).reset_index().drop(columns=['index'])
        w1 = reviews_full[(reviews_full['type'] == 'white') & (reviews_full['price'] <= 15.0)]
        df_w1 = pd.concat([df_w1, w1]).reset_index().drop(columns=['index'])
        w2 = reviews_full[(reviews_full['type'] == 'white') & (reviews_full['price'] > 15.0) & (reviews_full['price'] <= 35.0)]
        df_w2 = pd.concat([df_w2, w2]).reset_index().drop(columns=['index'])
        w3 = reviews_full[(reviews_full['type'] == 'white') & (reviews_full['price'] >= 35.0)]
        df_w3 = pd.concat([df_w3, w3]).reset_index().drop(columns=['index'])

        review_count += 2500



######################################### Review Generation From Scratch #########################################
def clean_reviews_generator(review):
    """
    For the review generator task, leave in punctuation and stopwords;
    -Remove foreign characters
    -Remove hyperlinks and special characters
    """
    clean_review = str(review) # ensure all reviews are string datatype
    clean_review = remove_foreign_characters(clean_review) # remove foreign characters
    clean_review = remove_numbers(clean_review)
    clean_review = clean_review.lower() # convert to lowercase
    clean_review = re.sub(r"!gif\(giphy.*\)", "", clean_review) # remove gifs
    clean_review = re.sub(r"\(https[^\s]*\)", " ", clean_review) # remove embedded hyperlinks
    clean_review = re.sub(r"https[^\s]*", " ", clean_review) # remove remaining hyperlinks
    clean_review = re.sub(r"amp;", "", clean_review)
    clean_review = re.sub(r"[\$\*'’\[\]#,%\^&\.\?\!<>]", "", clean_review)
    clean_review = re.sub(r"[/\-_+\\]", " ", clean_review)
    return clean_review

def fit_tokenizer(review_text, vocab_size=40000, oov_tok="<OOV>"):
    """
    Use keras.text Tokenizer to tokenize texts, adding a bos and eos token to indicate review start and finish. 
    """
    bos_token = "<BOS>"
    eos_token = "<EOS>"
    review_text = bos_token + " " + review_text + " " + eos_token
    # initialize tokenizer
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    # fit tokenizer
    tokenizer.fit_on_texts(review_text)
    return tokenizer

######################################### Review Generation GPT2 #########################################

def clean_text(text):
    """
    - remove any html tags (< /br> often found)
    - Keep only ASCII + Latin chars, digits and whitespaces
    - pad punctuation chars with whitespace
    - convert all whitespaces (tabs etc.) to single wspace
    """
    RE_PUNCTUATION = re.compile("([!?.,;-])")
    RE_TAGS = re.compile(r"<[^>]+>")
    RE_ASCII = re.compile(r"[^A-Za-zÀ-ž,.!?0-9 ]", re.IGNORECASE)
    RE_WSPACE = re.compile(r"\s+", re.IGNORECASE)
    text = re.sub(RE_TAGS, " ", text)
    text = re.sub(RE_ASCII, " ", text)
    text = re.sub(RE_PUNCTUATION, r" \1 ", text)
    text = re.sub(RE_WSPACE, " ", text)
    return text






