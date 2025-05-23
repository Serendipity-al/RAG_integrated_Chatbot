# RAG_integrated_Chatbot
This is a simple AI-powered chatbot built using Python, TensorFlow/Keras, and NLTK. The chatbot uses a trained neural network model (chatbot.h5) to classify intents based on user input and respond appropriately based on predefined intents in a JSON file.

<h1>Project Structure</h1>
RAG_Integrated_Chatbot/
│
├── chatbot.h5            # Trained neural network model
├── word.pkl              # Pickled list of vocabulary words
├── classes.pkl           # Pickled list of intent classes
├── intents.json          # Intents dataset used for training and responding
├── Chatbot.py            # Main Python script to run the chatbot
└── README.md             # Project documentation (this file)


<h1> How It Works </h1>
Tokenization & Lemmatization: User input is broken down into tokens and reduced to base form using NLTK.

Bag of Words: The input sentence is converted into a binary vector that the model can understand.

Intent Classification: The neural network predicts the most likely intent based on the input.

Response Selection: A response is randomly selected from the matching intent's predefined responses in intents.json.

<h1>Training the Model</h1>
If you don’t have the .pkl or .h5 files, you’ll need to train the model using your intents.json data. A separate train_chatbot.py script would be used to:

Preprocess the data

Train a neural network

Save the trained model and supporting files



Developed by Pari Jain
Feel free to connect or reach out on LinkedIn