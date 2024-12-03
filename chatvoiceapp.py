import nltk
import streamlit as st
import speech_recognition as sr
from googletrans import Translator
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

# Étape 2 : Télécharger les ressources NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Étape 3 : Charger le fichier texte contenant les questions et réponses
with open('assurance_automobile_faq.txt', 'r', encoding='utf-8') as f:
    lines = [line.strip() for line in f if line.strip()]

questions = []
reponses = []

# Extraction des questions et réponses
i = 0
while i < len(lines):
    if lines[i].startswith('Question:'):
        question = lines[i].replace('Question:', '').strip()
        if i + 1 < len(lines) and lines[i + 1].startswith('Réponse:'):
            reponse = lines[i + 1].replace('Réponse:', '').strip()
            questions.append(question)
            reponses.append(reponse)
            i += 2
        else:
            i += 1
    else:
        i += 1

assert len(questions) == len(reponses), "Le nombre de questions et de réponses ne correspond pas."

# Étape 4 : Fonction de prétraitement du texte
def preprocess(text):
    words = word_tokenize(text, language='french')
    words = [word.lower() for word in words if word.lower() not in stopwords.words('french') and word not in string.punctuation]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

questions_preprocessed = [preprocess(question) for question in questions]

# Étape 5 : Fonction pour obtenir la réponse la plus pertinente
def get_most_relevant_response(user_query):
    user_query_processed = preprocess(user_query)
    vectorizer = TfidfVectorizer().fit(questions_preprocessed + [user_query_processed])
    vectors = vectorizer.transform(questions_preprocessed + [user_query_processed])
    user_vector = vectors[-1]
    cosine_similarities = cosine_similarity(user_vector, vectors[:-1])
    most_similar_index = np.argmax(cosine_similarities)
    return reponses[most_similar_index]

# Étape 6 : Fonction de transcription vocale
def transcribe_speech(api_choice, speech_language, duration=60):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Parlez maintenant... (60 secondes maximum)")
        try:
            audio_text = r.listen(source, timeout=20, phrase_time_limit=duration)
            st.info("Transcription en cours...")
            if api_choice == "Google":
                text = r.recognize_google(audio_text, language=speech_language)
            else:
                return "API non supportée."
            return text
        except sr.UnknownValueError:
            return "Désolé, je n'ai pas compris."
        except sr.RequestError as e:
            return f"Erreur du service de reconnaissance vocale : {e}"
        except Exception as e:
            return f"Erreur : {str(e)}"

# Étape 7 : Fonction principale du chatbot vocal
def main():
    st.title("Chatbot Vocal Assurance Automobile")

    st.subheader("Hussein DIALLO")
    
    st.write("Posez une question en parlant ou en écrivant.")

    api_choice = st.selectbox("Choisissez l'API de reconnaissance vocale :", options=["Google"], index=0)
    speech_language = st.selectbox("Choisissez la langue :", options=["fr-FR", "en-US"], index=0)

    if 'response' not in st.session_state:
        st.session_state.response = None

    with st.form(key='question_form', clear_on_submit=True):
        user_query = st.text_input("Votre question :")
        submit_button = st.form_submit_button(label='Soumettre')

    if st.button("Parler maintenant"):
        user_query = transcribe_speech(api_choice, speech_language)
        st.write(f"**Vous avez dit :** {user_query}")
        st.session_state.response = get_most_relevant_response(user_query)

    if submit_button and user_query:
        st.write(f"**Question :** {user_query}")
        st.session_state.response = get_most_relevant_response(user_query)

    if st.session_state.response:
        st.markdown(f'**Réponse :** {st.session_state.response}')
        
if __name__ == "__main__":
    main()
