from flask import Flask, request, jsonify, render_template, session
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import string
from googletrans import Translator
from difflib import get_close_matches
import re

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

app = Flask(__name__)


app.secret_key = 'c84dac217816a52b78e8f27a0e3bed5c4fb9ed19ebe10587'

# Încărcarea datelor
data = {
    "questions": [
        "What is your return policy?",
        "How can I track my order?",
        "Do you offer international shipping?",
        "What payment methods do you accept?",
        "How can I contact customer service?",
        "What are your business hours?",
        "Do you have a physical store?",
        "Can I change my order after placing it?",
        "How long does delivery take?",
        "Do you offer gift wrapping services?",
        "What should I do if I receive a damaged item?",
        "Can I cancel my order?",
        "How do I reset my password?",
        "What is your privacy policy?",
        "Do you have a loyalty program?",
        "How do I use a discount code?",
        "Hi",
        "Hello",
        "Hey",
        "Can you help me?",
        "How are you?",
        "What can you do?",
        "Who are you?",
        "How can I place an order?",
        "How can I return a product?",
        "What are the steps to make a purchase?",
        "Can you tell me your working hours?",
        "What time do you open?",
        "What time do you close?",
        "How can I send a complaint?",
        "How do I contact support?",
        "What is the warranty period for products?",
        "Can I get a refund?",
        "Do you have a newsletter?",
        "How can I subscribe to the newsletter?",
        "What are the shipping charges?",
        "Do you ship to my country?",
        "How do I update my account information?",
        "What if I forget my account password?",
        "How can I delete my account?",
        "Can I get assistance with my order?",
        "What is your exchange policy?""Do you offer free shipping?",
        "Can I place an order over the phone?",
        "What is your average response time for customer inquiries?",
        "Are there any restrictions on international shipping?",
        "Do you have a mobile app?",
        "Is there a minimum order requirement?",
        "How do I check the status of my order?",
        "What is your customer satisfaction guarantee?",
        "Do you offer expedited shipping options?",
        "Can I change the shipping address after placing an order?"
        "Do you offer price matching?",
        "What is your policy on backordered items?",
        "Can I track my package without a tracking number?",
        "Do you have a satisfaction guarantee?",
        "What is your process for handling returns?",
        "Are there any restrictions on using discount codes?",
        "Can I return an item bought on sale?",
        "What is your policy on product exchanges?",
        "Do you offer installation services for your products?",
        "Can I schedule a delivery time?"
    ],
    "answers": [
        "You can return any item within 30 days of purchase.",
        "You can track your order using the tracking number provided in your confirmation email.",
        "Yes, we offer international shipping to many countries.",
        "We accept all major credit cards, PayPal, and bank transfers.",
        "You can contact customer service by phone or email at support@example.com.",
        "Our business hours are Monday to Friday, 9 AM to 6 PM.",
        "Yes, we have a physical store located at 123 Main Street.",
        "You can change your order within 1 hour of placing it by contacting customer service.",
        "Delivery usually takes 3-5 business days.",
        "Yes, we offer gift wrapping services for a small additional fee.",
        "If you receive a damaged item, please contact our customer service immediately.",
        "Yes, you can cancel your order within 1 hour of placing it.",
        "To reset your password, click on 'Forgot password' on the login page and follow the instructions.",
        "You can read our privacy policy on our website under 'Privacy Policy'.",
        "Yes, we have a loyalty program. You can sign up on our website.",
        "To use a discount code, enter it at checkout in the 'Discount Code' field.",
        "Hello! How can I assist you today?",
        "Hi there! How can I help you?",
        "Hey! How can I assist you?",
        "Of course! What do you need help with?",
        "I'm just a bot, but I'm here to help!",
        "I can assist you with common questions about our services.",
        "I'm a chatbot created to help answer your questions.",
        "You can place an order by adding items to your cart and proceeding to checkout.",
        "To return a product, please visit our returns page and follow the instructions.",
        "The steps to make a purchase are: select your items, add them to the cart, and proceed to checkout.",
        "Our working hours are Monday to Friday, 9 AM to 6 PM.",
        "We open at 9 AM from Monday to Friday.",
        "We close at 6 PM from Monday to Friday.",
        "You can send a complaint by contacting our support team via email.",
        "To contact support, email us at support@example.com or call us at our hotline.",
        "The warranty period for our products is typically 1 year.",
        "Yes, you can get a refund within 30 days of purchase.",
        "Yes, we have a newsletter. You can sign up on our website.",
        "To subscribe to our newsletter, enter your email in the subscription box on our website.",
        "Shipping charges vary based on location and order size. Check our shipping policy for details.",
        "Yes, we ship to many countries worldwide. Check our shipping policy for more details.",
        "To update your account information, go to 'My Account' and edit your details.",
        "If you forget your account password, click 'Forgot password' on the login page.",
        "To delete your account, contact our support team with your request.",
        "Yes, our support team can assist you with your order. Contact us for help.",
        "Our exchange policy allows you to exchange items within 30 days of purchase."
        "Yes, we offer free shipping on orders over $50.",
        "Yes, you can place an order over the phone by calling our sales team.",
        "Our average response time for customer inquiries is within 24 hours.",
        "There may be restrictions on certain items for international shipping due to customs regulations.",
        "Yes, we have a mobile app available for download on iOS and Android devices.",
        "There is no minimum order requirement for most purchases.",
        "You can check the status of your order by logging into your account and viewing your order history.",
        "We guarantee customer satisfaction. If you're not satisfied with your purchase, contact us for a resolution.",
        "Yes, we offer expedited shipping options for an additional fee.",
        "You can change the shipping address within 1 hour of placing your order by contacting customer service."
        "Yes, we offer price matching for identical items from competitors within 30 days of purchase.",
        "Our policy on backordered items is to notify customers of the delay and provide estimated restock dates.",
        "Unfortunately, tracking your package without a tracking number is not possible.",
        "Yes, we have a satisfaction guarantee. If you're not satisfied with your purchase, we'll make it right.",
        "To initiate a return, simply contact our customer service team and follow their instructions.",
        "Discount codes may have restrictions such as expiration dates or minimum purchase requirements. Please check the terms before use.",
        "Yes, you can return an item bought on sale within our standard return period.",
        "Our policy on product exchanges allows for exchanges within 30 days of purchase, subject to availability.",
        "We offer installation services for select products. Please check product descriptions or contact customer service for details.",
        "Unfortunately, we do not currently offer the option to schedule delivery times."
    ]
}

df = pd.DataFrame(data)

def expand_with_synonyms(text):
    words = text.split()
    expanded_text = set(words)
    for word in words:
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                expanded_text.add(lemma.name())
    return ' '.join(expanded_text)


# Funcție pentru preprocesarea textului
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    text = ' '.join([lemmatizer.lemmatize(word) for word in words if word not in stop_words])
    expanded_text = expand_with_synonyms(text)
    return expanded_text

df['processed_questions'] = df['questions'].apply(preprocess_text)

# Vectorizarea întrebărilor folosind TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['processed_questions'])

translator = Translator()

# Funcție pentru a găsi cea mai similară întrebare frecventă
def find_most_similar_question(user_question, lang):
    user_question_translated = translator.translate(user_question, dest='en').text
    user_question_processed = preprocess_text(user_question_translated)
    user_tfidf = vectorizer.transform([user_question_processed])
    similarities = cosine_similarity(user_tfidf, tfidf_matrix)
    most_similar_index = np.argmax(similarities)
    answer = df.iloc[most_similar_index]['answers']
    answer_translated = translator.translate(answer, dest=lang).text
    return answer_translated, df.iloc[most_similar_index]['questions']

@app.route('/')
def home():
    # Inițializează sesiunea pentru a păstra istoricul conversației
    session['conversation_history'] = []
    session['language'] = 'en'
    return render_template('index.html')

@app.route('/set_language', methods=['POST'])
def set_language():
    lang = request.json.get('language')
    session['language'] = lang
    return jsonify({'status': 'Language set to ' + lang})

correct_words = [
    "what", "how", "when", "where", "why", "who",
    "return", "track", "order", "contact", "business",
    "physical", "store", "change", "delivery", "gift",
    "damaged", "cancel", "reset", "privacy", "loyalty",
    "discount", "help", "place", "return", "purchase",
    "working", "open", "close", "complaint", "support",
    "warranty", "refund", "newsletter", "subscribe",
    "shipping", "ship", "update", "forget", "delete",
    "assistance", "exchange", "free", "price", "policy",
    "track", "package", "satisfaction", "process",
    "restriction", "restriction", "install", "schedule"
]

jokes = [
    "Why don't scientists trust atoms? Because they make up everything! 😄",
    "Why did the scarecrow win an award? Because he was outstanding in his field! 😄",
    "Why don’t skeletons fight each other? They don’t have the guts. 😄",
    "What do you call fake spaghetti? An impasta! 😄",
    "Why did the math book look sad? Because it had too many problems. 😄",
    "Why was the computer cold? It left its Windows open! 😄",
    "What did one ocean say to the other ocean? Nothing, they just waved. 😄",
    "Why can’t your nose be 12 inches long? Because then it would be a foot! 😄",
    "What do you get when you cross a snowman and a vampire? Frostbite. 😄",
    "Why don’t some couples go to the gym? Because some relationships don’t work out. 😄"
]

#Oana la cuvinte obsecene ar trebui sa ti afiseze mesajul ala
obscene_words = ["obscene1", "obscene2", "obscene3"]


#Oana aici am modificat vezi daca iti ia cu cuvinte gresite gen wht sa l recunoasca ca what sau ceva
@app.route('/get_answer', methods=['POST'])
def get_answer():
    user_question = request.json.get('question')
    lang = session.get('language', 'en')
    conversation_history = session.get('conversation_history', [])

    # Adaugă întrebarea utilizatorului la istoric
    conversation_history.append({'role': 'user', 'message': user_question})
    # Verificare mesaje licențioase
    if any(word in user_question.lower() for word in obscene_words):
        return jsonify({'answer': "The message contains licensed content and cannot be processed."})

        # Verifică dacă utilizatorul dorește o altă glumă
    if session.get('joke_mode', False) and re.search(r'\banother one\b', user_question, re.IGNORECASE):
        joke = np.random.choice(jokes)
        return jsonify({'answer': joke})

    # Răspunsuri pentru emoticoane specifice folosind regex
    if re.search(r'<3', user_question):
        session['joke_mode'] = False
        return jsonify({'answer': "I love you too <3"})
    if re.search(r':\(\(\(', user_question):
        session['joke_mode'] = False
        return jsonify({'answer': "Why are you sad? *sends virtual hugs*"})
    if re.search(r':\)+', user_question):
        session['joke_mode'] = False
        return jsonify({'answer': "I'm happy for you! :)))"})
    if re.search(r'\b(cute|adorable|sweet|nice)\b', user_question, re.IGNORECASE):
        session['joke_mode'] = False
        return jsonify({'answer': "Aww, thank you! You're awesome! 😊"})
    if re.search(r'\b(love|affection|care)\b', user_question, re.IGNORECASE):
        session['joke_mode'] = False
        return jsonify({'answer': "Sending you lots of love and positive vibes! ❤️"})
    if re.search(r'\b(friend|buddy|pal)\b', user_question, re.IGNORECASE):
        session['joke_mode'] = False
        return jsonify({'answer': "I'm here for you, buddy! 😊"})
    if re.search(r'\b(awesome|amazing|fantastic|great)\b', user_question, re.IGNORECASE):
        session['joke_mode'] = False
        return jsonify({'answer': "You're fantastic! Keep being awesome! 🌟"})
    if re.search(r'\b(thank you|thanks)\b', user_question, re.IGNORECASE):
        session['joke_mode'] = False
        return jsonify({'answer': "You're welcome! 😊"})
    if re.search(r'\b(happy|joy|delight)\b', user_question, re.IGNORECASE):
        session['joke_mode'] = False
        return jsonify({'answer': "I'm so glad to hear that! Keep smiling! 😊"})
    if re.search(r'\b(sad|unhappy|depressed)\b', user_question, re.IGNORECASE):
        session['joke_mode'] = False
        return jsonify({'answer': "I'm here for you. It's okay to feel sad sometimes. *sends virtual hugs*"})
    if re.search(r'\b(help|assist|support)\b', user_question, re.IGNORECASE):
        session['joke_mode'] = False
        return jsonify({'answer': "Of course! I'm here to help you. What do you need assistance with? 😊"})
    if re.search(r'\b(funny|joke|laugh)\b', user_question, re.IGNORECASE):
        session['joke_mode'] = True
        joke = np.random.choice(jokes)
        return jsonify({'answer': joke})
    if re.search(r'\b(tired|exhausted)\b', user_question, re.IGNORECASE):
        session['joke_mode'] = False
        return jsonify({'answer': "Take a deep breath and relax. You've got this! 🌟"})
    if re.search(r'\b(bored)\b', user_question, re.IGNORECASE):
        session['joke_mode'] = False
        return jsonify({'answer': "How about we play a game or I tell you a fun fact? 😊"})
    if re.search(r'\b(good night|gn)\b', user_question, re.IGNORECASE):
        session['joke_mode'] = False
        return jsonify({'answer': "Good night! Sweet dreams! 🌙"})
    if re.search(r'\b(good morning|gm)\b', user_question, re.IGNORECASE):
        session['joke_mode'] = False
        return jsonify({'answer': "Good morning! Have a fantastic day! ☀️"})
    if re.search(r'\b(hungry|starving)\b', user_question, re.IGNORECASE):
        session['joke_mode'] = False
        return jsonify({'answer': "Maybe you should grab a snack! 🍎"})
    if re.search(r'\b(thirsty)\b', user_question, re.IGNORECASE):
        session['joke_mode'] = False
        return jsonify({'answer': "Don't forget to stay hydrated! 💧"})
    if re.search(r'\b(angry|mad|furious)\b', user_question, re.IGNORECASE):
        session['joke_mode'] = False
        return jsonify({'answer': "Take a moment to calm down. Deep breaths can help. 🌬️"})

    # Corectează cuvântul greșit
    corrected_question = []
    for word in user_question.split():
        closest_match = get_close_matches(word.lower(), correct_words, n=1, cutoff=0.8)
        if closest_match:
            corrected_question.append(closest_match[0])
        else:
            corrected_question.append(word)
    corrected_question = ' '.join(corrected_question)

    answer, question_matched = find_most_similar_question(corrected_question, lang)

    # Adaugă răspunsul botului la istoric
    conversation_history.append({'role': 'bot', 'message': answer})
    session['conversation_history'] = conversation_history

    # Găsește contextul pentru întrebările de follow-up
    if user_question.lower() in ["how?", "why?", "when?", "where?", "what?"]:
        last_user_question = next((item['message'] for item in reversed(conversation_history) if item['role'] == 'user' and item['message'].lower() not in ["how?", "why?", "when?", "where?", "what?"]), None)
        if last_user_question:
            answer = f"You asked '{last_user_question}' before. {answer}"

    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)