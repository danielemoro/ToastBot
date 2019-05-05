import pandas as pd
import numpy as np
import string
from string import digits
import matplotlib.pyplot as plt
import re
import sklearn
from sklearn.cross_validation import train_test_split
import nltk
import matplotlib.pyplot as plt
from collections import Counter
import nltk
from nltk.translate.bleu_score import corpus_bleu
import os
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model
from keras.utils import plot_model
from keras.layers.merge import concatenate
import IPython
from tqdm import tqdm_notebook as tqdm
import pickle

def define_model(input_tokenizer, target_tokenizer, embedding_matrix_input, embedding_matrix_target):
    # define the encoder
    embedding_dim_input = embedding_matrix_input.shape[1]
    encoder_inputs = Input(shape=(None,))
    en_x =  Embedding(len(input_tokenizer), embedding_dim_input, weights=[embedding_matrix_input], trainable = False)(encoder_inputs)
    encoder = LSTM(50, return_state=True)
    encoder_outputs, state_h, state_c = encoder(en_x) #discard output and keep states
    encoder_states = [state_h, state_c]

    # define the decoder
    embedding_dim_output = embedding_matrix_input.shape[1]
    num_decoder_tokens = len(target_tokenizer)
    decoder_inputs = Input(shape=(None,))
    dex =  Embedding(num_decoder_tokens, embedding_dim_output, weights=[embedding_matrix_target], trainable = False)
    final_dex = dex(decoder_inputs)

    input_image = Input(shape=(None, 1000))
    y = input_image
    y = Model(inputs=input_image, outputs=y)

    combined = concatenate([final_dex, y.output])

    decoder_lstm = LSTM(50, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(combined, initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, y.input, decoder_inputs], decoder_outputs)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

    # compile the encoder and decoder models for prediction
    encoder_model = Model(encoder_inputs, encoder_states)
    decoder_state_input_h = Input(shape=(50,))
    decoder_state_input_c = Input(shape=(50,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    final_dex2= dex(decoder_inputs)
    input_image2 = Input(shape=(None, 1000))
    y2 = input_image2
    y2 = Model(inputs=input_image2, outputs=y2)
    combined = concatenate([final_dex2, y.output])

    decoder_outputs2, state_h2, state_c2 = decoder_lstm(combined, initial_state=decoder_states_inputs)
    decoder_states2 = [state_h2, state_c2]
    decoder_outputs2 = decoder_dense(decoder_outputs2)
    decoder_model = Model(
        [decoder_inputs, y.input] + decoder_states_inputs,
        [decoder_outputs2] + decoder_states2)

    return model, encoder_model, decoder_model

def decode_sequence(input_seq, input_img, encoder_model, decoder_model, input_tokenizer, target_tokenizer):
    reverse_input_char_index = dict( (i, char) for char, i in input_tokenizer.items())
    reverse_target_char_index = dict((i, char) for char, i in target_tokenizer.items())
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = target_tokenizer['START_']

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = []
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq, input_img] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence.append(sampled_char)

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '_END' or
           len(decoded_sentence) > 100):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]

    return decoded_sentence

def get_one_img(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    prediction_features = vgg19_model.predict(x)
    return prediction_features.ravel()

def user_input(input_words, img_feats, seq_len):
    input_seq = np.array([input_tokenizer[w] if w in input_tokenizer else 0 for w in input_words.split()] +
                         [0] * (seq_len - len(input_words.split()))).reshape(1, seq_len)
    input_img = np.array(img_feats.reshape(1,1,1000))
    inv_input_token_index = {v: k for k, v in input_tokenizer.items()}
    #print("raw input: [{}] {}".format(" ".join([inv_input_token_index[i] for i in input_seq[0]]), input_seq.shape))
    decoded_sentence = decode_sequence(input_seq, input_img, encoder_model, decoder_model, input_tokenizer, target_tokenizer)
    return " ".join(decoded_sentence).replace("START_ ", "").replace(" _END","")

def get_compliment(text, img_path):
    img = get_one_img(img_path)
    return user_input(text, img, 69)

# the vgg 19 model is used for getting the image features
vgg19_model = VGG19(weights='imagenet')
vgg19_model._make_predict_function()
vgg19_model = Model(inputs=vgg19_model.input, outputs=vgg19_model.get_layer('predictions').output)

# compliment generation model
input_tokenizer, target_tokenizer, input_embedding_matrix, output_embedding_matrix = pickle.load(open("model.pkl", "rb"))
model, encoder_model, decoder_model = define_model(input_tokenizer, target_tokenizer,
                                                   input_embedding_matrix, output_embedding_matrix)
model.load_weights("model.h5")
