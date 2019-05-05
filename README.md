# ToastMe
_Daniele Moro_ and _Oghenemaro Anuyah_

CS 497 Applied Deep Learning Final Project

This is a bot that automatically generates personal compliments based on a user's image and written emotional state. It uses a seq2seq model with an LSTM and word-level embeddings.

For more details, see `FinalReport.pdf`

# Installation
1. Set up a `Python 3` environment with the packages found in `env.txt`
2. Download `glove.6B.zip` from the [Glove site](https://nlp.stanford.edu/projects/glove/) and place the files in the root directory
2. Run `RedditDownload.ipynb` to download the data for training and extract the visual features from the images.
3. Run `ToastBot.ipynb` to train and evaluate a model that uses both the visual and textual features. You can also evaluate our existing model by loading `model.h5` before evaluation.
4. (Optional) run `ToastBot_images.ipynb` and `ToastBot_text.ipynb` to train and evaluate models that only use either textual or image data respectively.

# Web Server
To run the web server for easy interaction with the compliment generator, run `ToastBot_server.py`, and it will start up a Flask server on your localhost. The server will use `ToastBot_predict.py` to run the compliment generation, so make sure this file is loading the appropriate model.
