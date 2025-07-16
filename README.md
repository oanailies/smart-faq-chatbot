# Smart FAQ Chatbot

This repository contains the implementation of a smart chatbot system designed to recognize frequently asked questions (FAQ) for an e-commerce platform. The system is built using Flask and supports automatic response generation to user queries by leveraging natural language processing techniques such as TF-IDF and cosine similarity.

## Project Overview

The chatbot is intended to improve user experience by providing immediate and accurate answers to frequently asked questions. It can be integrated into any online store and is designed to reduce the need for human customer service intervention.

## Key Features

- Web interface built with Flask for direct interaction.
- Predefined question-answer database in JSON format.
- TF-IDF vectorization and cosine similarity used for matching user questions with the most relevant existing question.
- Automatic language translation using Google Translate API to support multilingual users.
- Preprocessing pipeline for cleaning, normalization, and tokenization of text.
- Handling of out-of-context inputs such as jokes, emojis, and small talk.
- Detection and filtering of inappropriate or obscene language.

## How It Works

1. The user accesses the web interface and types a question.
2. The system preprocesses and (if needed) translates the input to English.
3. The question is vectorized using TF-IDF and compared to existing questions using cosine similarity.
4. The most relevant response is identified and translated back to the user's preferred language.
5. The answer is displayed on the interface.

## Technologies Used

- Python 3
- Flask
- Scikit-learn (TF-IDF, cosine similarity)
- Google Translate API
- HTML, CSS, JavaScript

## Future Improvements

- Continuous learning based on user feedback.
- Integration with external messaging platforms (e.g., Facebook Messenger, WhatsApp).
- Sentiment analysis and named entity recognition.
- Improved multilingual support.
- Enhanced security and privacy for user data.

## Team

This project was developed as part of a university group assignment at the Faculty of Automation and Computer Science.

- Ilies Oana-Elena – Responsible for response matching logic, synonym handling, and similarity computation.
- Flutur Florina – Responsible for UI development, multilingual integration, and moderation system.
- Anton Mihaela-Camelia – Responsible for data preparation, text vectorization, and humor/emoji handling.

## License

This project is developed for educational purposes only and is not intended for commercial use.

