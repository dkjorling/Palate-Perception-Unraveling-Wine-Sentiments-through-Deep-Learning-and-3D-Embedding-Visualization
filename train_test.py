import tensorflow as tf
import pandas as pd
import numpy as np
import random
import pickle
import torch
import torch.nn.functional as F
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
from tqdm import tqdm, trange


torch.manual_seed(27) # set torch random seed

# my imports
import data_clean as dc
################### Classification Modeling  ############################
#### Preprocessing ####
def get_labels(review_df):
    """
    Classify review as positive or negative using 4.0 as cutpoint and return binary np array 
    """
    labels = []
    ratings = review_df.rating
    for r in ratings:
        if r >= 4.0:
            labels.append(1)
        else:
            labels.append(0)
    return np.array(labels)


def get_features(train_text, val_text, vocab_size=30000, oov_tok="<OOV>", max_length=250, padding='post', trunc_type='post'):
    """
    Take in training and validation text sets and return tokenized training/validation sequences padded to max length. 
    """
    # load tokenizer
    with open('data/tokenizer_30k_class.pickle', 'rb') as file:
        tokenizer = pickle.load(file)
    # fit on training 
    tokenizer.fit_on_texts(train_text)
    word_index = tokenizer.word_index
    
    # generate and pad training sequences
    train_seq = tokenizer.texts_to_sequences(train_text)
    train_pad = pad_sequences(train_seq,
                              maxlen=max_length,
                              padding=padding,
                              truncating=trunc_type)
    
    val_seq = tokenizer.texts_to_sequences(val_text)
    val_pad = pad_sequences(val_seq,
                           maxlen=max_length,
                          padding=padding,
                          truncating=trunc_type)
    
    return train_pad, val_pad

#### Modeling ####
class TransformerBlock(layers.Layer):
    """
    Code From: https://keras.io/examples/nlp/text_classification_with_transformer/
    - Create Transformer Block layer class
    """
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(layers.Layer):
    """
    Code From: https://keras.io/examples/nlp/text_classification_with_transformer/
    - Create Embedding and Positional layer class for Transformer
    """
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
    
    
def train_transformer_classifier(feats, labels, embed_dim, num_heads, ff_dim, max_len, voocab_size, dropout_rate, epochs):
    """
    Take in features and labels and train TensorFlow Transformer for review classification
    - Stratified KFold used due to imbalance of positive/negative reviews
    """
    lr_scheduler_callback = LearningRateScheduler(lr_scheduler)

    inputs = layers.Input(shape=(max_len,)) # input layer
    embedding_layer = TokenAndPositionEmbedding(max_len, vocab_size, embed_dim) # initialize embedding layer class
    x = embedding_layer(inputs) # add embedding layer
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim) # initialize transformer layer class
    x = transformer_block(x) # add transformer layer
    x = layers.GlobalAveragePooling1D()(x) # average pooling layer
    x = layers.Dropout(dropout_rate)(x) # add dropout layer
    x = layers.Dense(ff_dim, activation="relu")(x) # add feed-forward layer
    x = layers.Dropout(dropout_rate)(x) # add another dropout layer
    outputs = layers.Dense(1, activation="sigmoid")(x) # output layer use sigmoid activation to push to 0 or 1


    accuracy_scores = []
    loss_scores = []

    for kfold, (train, test) in enumerate(StratifiedKFold(n_splits=4, shuffle=True, random_state=27).split(feats, labels)):
        tf.keras.backend.clear_session()

        # Make sure equally balanced positive/negative reviews in both train/test
        positive_indices = np.where(labels[train] == 1)[0]
        negative_indices = np.where(labels[train] == 0)[0]

        num_samples = min(len(positive_indices), len(negative_indices)) # undersample positive reviews
        np.random.shuffle(positive_indices)
        np.random.shuffle(negative_indices)

        balanced_indices = np.concatenate([positive_indices[:num_samples], negative_indices[:num_samples]])
        np.random.shuffle(balanced_indices)

        feats_balanced = feats[train][balanced_indices]
        labels_balanced = labels[train][balanced_indices]

        model_transformer = keras.Model(inputs=inputs, outputs=outputs)
        model_transformer.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-4),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # optional load pre-trained weights if starting from end of previous subset
        # model_transformer.load_weights('drive/My Drive/transformer_w1_base.h5')

        h = model_transformer.fit(feats_balanced,
                              labels_balanced,
                              epochs=epochs,
                              validation_data=(feats[test], labels[test]),
                              callbacks=[lr_scheduler_callback])

        loss, accuracy = model_transformer.evaluate(feats[test], labels[test])
        accuracy_scores.append(accuracy)
        loss_scores.append(loss)
        
    return h, accuracy_scores, loss_scores

################### Review Generation From Scratch ############################
#### Preprocessing ####
def generate_expanding_sequences_gpt(clean_reviews, tokenizer, max_length=25, vocab_size=30000):
    """
    Takes in reviews that have been cleaned and returns n-gram sequences padded to max_length parameter
    """
    max_length = tokenizer.model_max_length
    vocab_size = tokenizer.vocab_size 
    input_sequences = []
    for review in clean_reviews:
        token_list = tokenizer(review)['input_ids'] # tokenize review
        token_list = token_list[:max_length+1] # truncate
        for i in range(3, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            padded = tokenizer.batch_encode_plus(
                                            n_gram_sequence,
                                            padding=True,  # Enable padding
                                            truncation=True,  # Enable truncation if needed
                                            max_length=max_length+1,  # Specify the maximum length of the sequences
                                            return_tensors="tf"  # Return TensorFlow tensors
                                        )
            input_sequences.append(padded)
    feats, labels = input_sequences[:,:-1], input_sequences[:,-1]
    #labels = tf.keras.utils.to_categorical(labels, num_classes=vocab_size)
    return feats, labels    

#### Modeling ####
def checkpoint_callback(checkpoint_path, save_freq):
    """
    Saves TensorFlow model weights after every save_freq steps
    """
    checkpoint_callback = ModelCheckpoint(checkpoint_path, save_weights_only=True, save_freq=(save_freq))
    return checkpoint_callback

def lr_scheduler(epoch, lr):
    """
    Learning rate scheduler helper function that initites at initial_lr and decays at rate
    """
    initial_lr = 0.005
    rate=2.0
    new_lr = initial_lr / (rate ** epoch)
    print(f"Learning rate: {new_lr}")
    return new_lr

def lr_scheduler_callback(lr_scheduler):
    """
    Initiate learning rate scheduler based on specifications set in lr_scheduelr function
    """
    lr_callback = LearningRateScheduler(lr_scheduler)
    return lr_callback

def build_gpt(vocab_size, max_length, num_heads, embed_dim, ff_dim, num_layer=2):
    """
    Build small Transformer model for text generation from scratch task
    - Num layers determines how many transformer blocks to iterate through in model
    """
    inputs = layers.Input(shape=(max_length,))
    embedding_layer = TokenAndPositionEmbedding(max_length, vocab_size+1, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    for _ in range(num_layers):
        x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(20, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(vocab_size+1, activation="softmax")(x) # softmax activate logits with size vocab +1 
    #print(outputs.shape)
    model = keras.Model(inputs=inputs, outputs=outputs) 
    return model

def data_generator(chunksize=500, batch_size=64, max_length=25, vocab_size=30000):
    """
    Initiate generator function used to train model so that memory is not over-allocated
    """
    while True:
        data_reader = pd.read_csv("drive/My Drive/pos_en.csv", index_col=0, chunksize=chunksize) # initiate reader
        # initiate tokenizer
        bos_token = "<BOS> "
        eos_token = "<EOS>"
        with open("drive/My Drive/tokenizer_30k.pickle", 'rb') as handle:
            tokenizer = pickle.load(handle)
        for chunk in data_reader:
            chunk['review_text'] = chunk.review_text.apply(clean_reviews_generator) # clean text
            review_text = bos_token + " " + chunk.review_text + " " + eos_token # format text w bos/eos tokens
            # generate feats and labels in chunk
            feats, labels = generate_expanding_sequences(review_text,
                                                          tokenizer=tokenizer,
                                                          max_length=max_length,
                                                          vocab_size=vocab_size)
            # count samples, get indices and shuffle
            num_samples = feats.shape[0]
            indices = np.arange(num_samples)
            np.random.shuffle(indices)

            for i in range(0, num_samples, batch_size):
                # shuffle batch order
                batch_indices = indices[i:i+batch_size]
                batch_feats = feats[batch_indices]
                batch_labels = labels[batch_indices]
                # feed model one batch at a time
                yield batch_feats, batch_labels

def train_generator_model(initial_learning_rate, vocab_size, max_length, num_heads, embed_dim, ff_dim, num_layers, epochs, batch_size, steps_per_epoch):
    """
    Train the Transformer to guess the next word in the sequence with the end goal to be able to write reviews from scratch.
    """
    # set path for model checkpoint
    checkpoint_path = 'drive/My Drive/weights25pos3.{epoch:02d}.h5'
    checkpoint_callback = ModelCheckpoint(checkpoint_path, save_weights_only=True, save_freq=(387399))

    # initiate model
    model = build_gpt(vocab_size,
                      max_length-1,
                      num_heads,
                      embed_dim,
                      ff_dim,
                      num_layers)
    # compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=initial_learning_rate),
                  metrics=['accuracy'])
    # load weights if starting from a checkpoint
    # model.load_weights("drive/My Drive/weights25pos2.01.h5")
    train_generator = data_generator(batch_size=batch_size, max_length=max_length, vocab_size=vocab_size) # initiate generator
    h = model.fit(train_generator, epochs=epochs, verbose=1, steps_per_epoch=steps_per_epoch, callbacks=[checkpoint_callback])
    
    return h


################### Review Generation Fine Tune GPT2 ############################
#### Preprocessing ####

class WineReviews(Dataset):
    """
    Create wine reviews dataset that returns inputs and attention mask to feed to GPT2 model.
    Note: Reviews must already contain specific bos tokens before passing to this class
    """
    def __init__(self, generator, tokenizer, gpt2_type="gpt2", max_length=1024):
        self.generator = generator
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.input_ids = []
        self.attn_masks = []

        txt_list = next(self.generator)
        for txt in txt_list:
            encodings_dict = tokenizer(txt + " <|endoftext|>", truncation=True, max_length=self.max_length, padding="max_length")
            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]
    
def data_generator_gpt2(chunksize=20000):
    """
    Function to generate chunks of data used for PyTorch GPT2LMHeadModel training
    """
    for chunk in pd.read_csv("drive/My Drive/reviews_base.csv", index_col=0, chunksize=chunksize):
        chunk = chunk['review_text']
        yield chunk

def format_time(elapsed):
    """
    Format time for training loop
    """
    return str(datetime.timedelta(seconds=int(round((elapsed)))))

def train_gpt2_generator(epochs, learning_rate, warmup_steps, epsilon, total_len, batch_size):
    """
    Fine-tune pre-trained GPT2 model to be able to generate reviews in the style of Vivino users
        Parameters:
        - epochs: number of epochs to train for
        - learning_rate: initial learning rate to use
        - warmup_steps: learning rate remains constant through warmup steps
        - epsilon: tuning parameter for optimizer
        - total_len: size of training data
        - batch_size: size of each batch fed to model
    """
    # Load the GPT tokenizer.
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<RED-NEG>', eos_token='<|endoftext|>')
    start_tokens = ["<RED-POS>", "<WHT-POS>", "<WHT-NEG>"]
    tokenizer.add_tokens(start_tokens, special_tokens=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    #instantiate the model
    configuration = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)
    model = GPT2LMHeadModel.from_pretrained("gpt2", config=configuration)
    # resize embedding layer to include custom tokens
    model.resize_token_embeddings(len(tokenizer))
    # ensure model using GPU
    device = torch.device("cuda")
    model.cuda()
    
    # set seed
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # initiate optimizer
    optimizer = AdamW(model.parameters(),
                  lr = learning_rate,
                  eps = epsilon
                )
    
    # initiate lr scheduler that decays linearly
    total_steps =  int(total_len / batch_size * epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = warmup_steps,
                                                num_training_steps = total_steps)
    
    # set start time and model save path
    total_t0 = time.time()
    save_path = "drive/My Drive/model_4bos.pt"
    save_every = 2000
    sample_every = 100 # produce an output every sample_every iterations
    training_stats = []
    batch_losses = []
    
    model = model.to(device)
    num_chunks = int(total_len // 20000) # determine total number of generator iterations
    
    ########### BEGIN TRAINING ###########
    # iterate over epochs
    for epoch_i in range(epochs):
        # initiate data generator
        generator = data_generator_gpt2()
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        t0 = time.time()
        total_train_loss = 0
        model.train()
        
        # iterate through each chunk
        for i in range(num_chunks):
            print('======== Chunk {:} / {:} ========'.format(i + 1, num_chunks))
            
            # create dataloader object to feed to model
            dataset = WineReviews(generator, tokenizer=tokenizer, gpt2_type="gpt2")
            train_dataloader = DataLoader(
                        dataset,  # The training samples.
                        sampler = RandomSampler(dataset), # Select batches randomly
                        batch_size = batch_size # Trains with this batch size.
                    )
            # iterate through each batch
            for step, batch in enumerate(train_dataloader):

                b_input_ids = batch[0].to(device)
                b_labels = batch[0].to(device)
                b_masks = batch[1].to(device)

                model.zero_grad()
                
                # Feed-Forward
                outputs = model(  b_input_ids,
                                  labels=b_labels,
                                  attention_mask = b_masks,
                                  token_type_ids=None
                                )

                loss = outputs[0] # calculate loss and append
                batch_loss = loss.item()
                batch_losses.append(batch_loss)
                total_train_loss += batch_loss
                
                # if save_every, save model to path
                if step % save_every == 0:
                    torch.save(model.state_dict(), save_path)

                # Get sample every x batches.
                if step % sample_every == 0 and not step == 0:
                    print(np.mean(batch_losses[-100:]))
                    elapsed = format_time(time.time() - t0)
                    print('  Batch {:>5,}  of  {:>5,}. Loss: {:>5,}.   Elapsed: {:}.'.format(step, len(train_dataloader), batch_loss, elapsed))

                    model.eval()

                    sample_outputs = model.generate(
                                            bos_token_id=tokenizer.bos_token_id,
                                            do_sample=True,
                                            top_k=50,
                                            max_length = 200,
                                            top_p=0.95,
                                            pad_token_id=tokenizer.pad_token_id,
                                            num_return_sequences=1
                                        )
                    for i, sample_output in enumerate(sample_outputs):
                          print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))

                    model.train()
                
                # backward pass 
                loss.backward()
                optimizer.step()
                scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))
        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Training Time': training_time,
            }
        )
        
        return training_stats, batch_losses
