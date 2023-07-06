import re
import pandas as pd
import pickle
import tensorflow as tf
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from sklearn.metrics.pairwise import cosine_similarity

import data_clean as dc


def get_token_dict(text_col):
    """
    - Tokenize each data subset and sort by term frequency
    - This function will be used to identify descriptors, wine types and positive/negative words
    """
    eng_stopwords = stopwords.words('english') 
    token_dict = {}
    for t in text_col:
        tokenized = word_tokenize(t)
        no_stop = [x for x in tokenized if x not in eng_stopwords]
        for t2 in no_stop:
            if t2 in token_dict.keys():
                token_dict[t2] += 1
            else:
                token_dict[t2] = 1
                
    sorted_token_dict = sorted(token_dict.items(), key=lambda x:x[1], reverse=True)
    sorted_token_dict = [x for x in sorted_token_dict if len(re.findall(r"[a-z]+", x[0])) > 0]
    
    return sorted_token_dict


def generate_review_scratch(model, max_length=25):
    """
    This function generates reviews beginning with the bos token up to max length.
    - Note that this function is valid only for small transformer built from scratch
    """
    with open("drive/My Drive/tokenizer_30k.pickle", 'rb') as handle:
        tokenizer = pickle.load(handle)
    seed_text = "bos"
    while True:
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_length-1, padding='pre')
        probs = model.predict(token_list, verbose=0)

        pred = tf.random.categorical(tf.math.log(probs), num_samples=1).numpy()[0][0]
        while pred == 1:
            pred = tf.random.categorical(tf.math.log(probs), num_samples=1).numpy()[0][0]
        if pred != 0:
            output_word = tokenizer.index_word[pred]
            seed_text += " " + output_word
            if output_word == "eos":
                seed_text = seed_text.split()
                seed_text = seed_text[1:-1]
                seed_text[0] = seed_text[0].capitalize()
                seed_text = " ".join(seed_text)
                print(seed_text)
                return
            
def load_review_generator():
    """
    Loads Fine-tuned GPT2 Model for review generation
    """
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<RED-NEG>', eos_token='<|endoftext|>')
    tokenizer.pad_token = tokenizer.eos_token
    start_tokens = ["<RED-POS>", "<WHT-POS>", "<WHT-NEG>"] # add the three other custom start tokens
    tokenizer.add_tokens(start_tokens, special_tokens=True)
    configuration = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)
    review_generator = GPT2LMHeadModel.from_pretrained("gpt2", config=configuration)
    review_generator.resize_token_embeddings(len(tokenizer))
    review_generator.load_state_dict(torch.load("drive/My Drive/model_4bos.pt", map_location=torch.device('cpu')))
    return review_generator

def clean_generation(review):
    """
    Format reviews generted by the model
    - Format contractions 
    - Format decimal points for numbers
    """
    clean = re.sub(r"\bI\sm\b", "I'm", review)
    clean = re.sub(r"\bi\sm\b", "I'm", clean)
    clean = re.sub(r"\bI\sve\b", "I've", clean)
    clean = re.sub(r"\bi\sve\b", "I've", clean)
    clean = re.sub(r"didn\st\b", "didn't", clean)
    clean = re.sub(r"Didn\st\b", "Didn't", clean)
    clean = re.sub(r"(\d{1})\.\s(\d{1,2})", r"\1.\2", clean)
    clean = re.sub(r"\bit\ss", "it's", clean)
    clean = re.sub(r"It\ss", "It's", clean)
    clean = re.sub(r"I\sll", "I'll", clean)
    clean = re.sub(r"doesn\st\b", "doesn't", clean)
    clean = re.sub(r"Doesn\st\b", "Doesn't", clean)
    clean = re.sub(r"\bdon\st\b", "don't", clean)
    clean = re.sub(r"Don\st\b", "Don't", clean)
    clean = re.sub(r"\bwon\st\b", "won't", clean)
    clean = re.sub(r"\bWon\st\b", "Won't", clean)
    clean = re.sub(r"wasn\st", "wasn't", clean)
    clean = re.sub(r"Wasn\st", "Wasn't", clean)
    clean = re.sub(r"\bcan\st\b", "can't", clean)
    clean = re.sub(r"\bCan\st\b", "Can't", clean)
    clean = re.sub(r"there\ss\b", "there's", clean)
    clean = re.sub(r"There\ss\b", "There's", clean)
    clean = re.sub(r"couldn\st\b", "couldn't", clean)
    clean = re.sub(r"shouldn\st\b", "shouldn't", clean)
    clean = re.sub(r"wouldn\st\b", "wouldn't", clean)
    clean = re.sub(r"\bit\sll\b", "it'll", clean)
    clean = re.sub(r"\bIt\sll\b", "It'll", clean)
    clean = re.sub(r"\byou\sd\b", "you'd", clean)
    clean = re.sub(r"\bYou\sd\b", "You'd", clean)

    return clean

def generate_review(kind='red_pos', rand=False):
    """
    Generate reviews with the option to select one of four types:
        1) Positive Red Wine 2) Negative Red Wine 3) Positive White Wine 4) Negative White Wine
    - If rand set to True, generator randomly selects which review type to write
    """
    kinds = ['red_neg', 'red_pos', 'white_pos', 'white_neg']
    bos_dict = {
        'red_neg': 50257,
        'red_pos': 50258,
        'white_pos': 50259,
        'white_neg': 50260
    }
    if kind not in kinds: # check to make sure kind is valid
        raise ValueError("Invalid kind; choose one of: red_pos, red_neg, white_pos, white_neg")
    if rand:
        kind = random.sample(kinds, 1)[0]
    review_generator = load_review_generator()
    review_generator.eval()
    bos_id = bos_dict[kind]
    output = review_generator.generate(
                            bos_token_id=bos_id,
                            do_sample=True,
                            top_k=50,
                            max_length=150,
                            top_p=0.95,
                            pad_token_id=tokenizer.pad_token_id,
                            num_return_sequences=1
                        )
    output = tokenizer.decode(output[0], skip_special_tokens=True)
    output = clean_generation(output)
    if kind != 'red_neg':
        output = output.split(" ")
        output = output[1:]
        output = " ".join(output)
    print(output)
    return kind, output

def drop_bos(text):
    """
    Formate real reviews back by removing bos token
    """
    clean = text.split(" ")
    clean = clean[1:]
    clean = " ".join(clean)
    return clean

def clean_punct(text):
    """
    Format real reviews back by stripping space before punctuation
    """
    clean = re.sub(r"\s+,", ",", text)
    clean = re.sub(r"\s+\.", ".", clean)
    clean = re.sub(r"\s+\!", "!", clean)
    clean = re.sub(r"\s+\?", "?", clean)
    return clean

def get_descriptive_word_scores(category='w1'):
    """
    Get average positive and negative cosine similarity scores for descriptors and wine types in category
    """
    # load word dict
    with open('data/words.json', 'r') as f:
        word_dict = json.load(f)
    # load pos and neg words
    pos_words = word_dict['pos_words']
    neg_words = word_dict['neg_words']
    
    # set list of descriptive words and wine type based on category
    if category[0] == 'w':
        descriptive_words = word_dict['white_words']
        wine_types = word_dict['white_types']
    else:
        descriptive_words = word_dict['red_words']
        wine_types = word_dict['red_types']
        
    # load embeddings into csv
    base = 'data/word_vecs/'
    word_file = base + 'meta_{}.tsv'.format(category)
    embedding_file = base + 'vecs_{}.tsv'.format(category)
    words = pd.read_csv(word_file, sep='\t', header=None)
    embeddings = pd.read_csv(embedding_file, sep='\t', header=None)
    embeddings.index = list(words[0].values)
    
    # set positive and negative word matrices
    pos_reference = embeddings.loc[pos_words].values
    neg_reference = embeddings.loc[neg_words].values
    
    # get avg cosine similarity for types
    pos_wine_type = []
    neg_wine_type = []
    for var in wine_types:
        vec = embeddings.loc[var].values.reshape(1, -1)
        pos_scores = cosine_similarity(vec, pos_reference)
        pos_wine_type.append(np.mean(pos_scores))
        neg_scores = cosine_similarity(vec, neg_reference)
        neg_wine_type.append(np.mean(neg_scores))
    
    pos_descriptor = []
    neg_descriptor = []
    drop_descriptor = []
    for word in descriptive_words:
        try:
            vec = embeddings.loc[word].values.reshape(1, -1)
            pos_scores = cosine_similarity(vec, pos_reference)
            pos_descriptor.append(np.mean(pos_scores))
            neg_scores = cosine_similarity(vec, neg_reference)
            neg_descriptor.append(np.mean(neg_scores))
        except:
            print("Word not in subset")
            drop_descriptor.append(word)
    
    # drop words not in set
    for word in drop_descriptor:
        descriptive_words.remove(word)
    # create a dataframes for both wine type and descriptive words and return
    wine_type_sentiment = pd.DataFrame(data={'pos':pos_wine_type,
                                             'neg':neg_wine_type},
                                       index=wine_types)   
    descriptor_sentiment = pd.DataFrame(data={'pos':pos_descriptor,
                                             'neg':neg_descriptor},
                                       index=descriptive_words)
    
    # add columns to indicate data subset and sort on positive
    wine_type_sentiment['subset'] = category
    wine_type_sentiment = wine_type_sentiment.sort_values('pos', ascending=False)
    descriptor_sentiment['subset'] = category
    descriptor_sentiment = descriptor_sentiment.sort_values('pos', ascending=False)
    
    return wine_type_sentiment, descriptor_sentiment

def plot_top_n_words(color='red', cat=1, kind='des', pos_neg='pos', n=20):
    """
    Plot top n words by positive or negative average cosine simlarity in descending order
    Parameters
    -color: red or white wine
    -cat: 1, 2, 3 for price ranges (cheap, medium, expensive)
    -pos_neg: positive or negative cosine similarity
    -n: number of words to plot
    """
    plt.rcParams['font.family'] = 'Helvetica'
    file = "data/visualizations/" + color + "_" + kind + ".csv"
    df = pd.read_csv(file, index_col=0)
    col = str('{}_{}{}'.format(pos_neg, color[:1], str(cat)))
    df = df.sort_values(col, ascending=False)
    # plot figure
    sns.set_style("ticks")
    plt.figure(figsize=(10, 6))

    # Create a color palette with a gradient of red colors
    if color=='red':
        palette = sns.color_palette("Reds_r", n)
    else:
        palette = sns.color_palette("YlOrBr", n)
    
    # automate title
    if pos_neg == 'pos':
        p = 'Positive'
    else:
        p = 'Negative'
    if kind == 'des':
        k = 'Descriptors'
    else:
        k = 'Wine Types'
    if cat == 1:
        r = '< \$15'
    elif cat == 2:
        r = '\$15-$35'
    else:
        r = '\$35-$80'
    c = color.capitalize()
    title = "Top {} Polarity {}: {} {} Wine".format(p, k, r, c)
    sns.barplot(x=list(df.index)[:n], y=df[col][:n], palette=palette)
    plt.ylabel('Polarity Score')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.show()