#%%
# Gerekli kütüphanelerin eklenmesi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, LSTM, GRU, Dense
#%%
# Veri setinin yüklenmesi
data = pd.read_csv('mtsamples.csv')
#%%
# Fonksiyon: Cümle ve kelime sayısını hesapla
def get_sentence_word_count(text_list):
    sent_count = 0
    word_count = 0
    vocab = {}
    
    for text in text_list:
        sentences = sent_tokenize(str(text).lower())
        sent_count += len(sentences)
        
        for sentence in sentences:
            words = word_tokenize(sentence)
            
            for word in words:
                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 1
    
    word_count = len(vocab.keys())
    return sent_count, word_count
#%%
# Veri kümesini yükle
clinical_text_df = pd.read_csv("mtsamples.csv")

# NaN transkripsiyonları temizle
clinical_text_df = clinical_text_df[clinical_text_df['transcription'].notna()]

# Kategori sayısı 50'nin altında olan transkripsiyonları temizle
filtered_data_categories = clinical_text_df.groupby('medical_specialty').filter(lambda x: x.shape[0] > 50)
#%%
# Veri ön işleme ve temizleme
def clean_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ''.join([w for w in text if not w.isdigit()])
    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub('', text)
    return text

def lemmatize_text(text):
    wordlist = []
    lemmatizer = WordNetLemmatizer()
    sentences = sent_tokenize(text)

    intial_sentences = sentences[:1]
    final_sentences = sentences[-2:-1]

    for sentence in intial_sentences:
        words = word_tokenize(sentence)
        wordlist.extend(lemmatizer.lemmatize(word) for word in words)

    for sentence in final_sentences:
        words = word_tokenize(sentence)
        wordlist.extend(lemmatizer.lemmatize(word) for word in words)

    return ' '.join(wordlist)

filtered_data_categories['transcription'] = filtered_data_categories['transcription'].apply(lemmatize_text)
filtered_data_categories['transcription'] = filtered_data_categories['transcription'].apply(clean_text)
#%%
# Veri kümesini eğitim ve test setlerine ayırma
X = filtered_data_categories['transcription']
y = filtered_data_categories['medical_specialty']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#%%
# TF-IDF vektörleştirme
vectorizer_tfidf = TfidfVectorizer(sublinear_tf=True, encoding='utf-8', decode_error='ignore')
X_train_tfidf = vectorizer_tfidf.fit_transform(X_train)
X_test_tfidf = vectorizer_tfidf.transform(X_test)
#%%
# SMOTE ile dengesiz sınıfları dengeleme
smote = SMOTE(random_state=42)
X_train_smote_tfidf, y_train_smote_tfidf = smote.fit_resample(X_train_tfidf, y_train)
#%%
# Naive Bayes
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_smote_tfidf, y_train_smote_tfidf)
y_pred_nb_tfidf = nb_classifier.predict(X_test_tfidf)
#%%
# Random Forest
rf_classifier = RandomForestClassifier(random_state=1)
rf_classifier.fit(X_train_smote_tfidf, y_train_smote_tfidf)
y_pred_rf_tfidf = rf_classifier.predict(X_test_tfidf)
#%%
# XGBoost
xgb_classifier = XGBClassifier(random_state=1)
# Encode labels using LabelEncoder
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train_smote_tfidf)
xgb_classifier.fit(X_train_smote_tfidf, y_train_encoded)
# Ensure that classes match
print("Expected Classes:", xgb_classifier.classes_)
y_pred_xgb_tfidf = xgb_classifier.predict(X_test_tfidf)
#%%
# LightGBM
lgb_classifier = lgb.LGBMClassifier(random_state=1)
lgb_classifier.fit(X_train_smote_tfidf, y_train_smote_tfidf)
y_pred_lgb_tfidf = lgb_classifier.predict(X_test_tfidf)
#%%
# Çıktıları CSV dosyalarına kaydetme
pd.DataFrame(y_pred_nb_tfidf).to_csv('y_pred_nb_tfidf.csv', index=False)
pd.DataFrame(y_pred_rf_tfidf).to_csv('y_pred_rf_tfidf.csv', index=False)
pd.DataFrame(y_pred_xgb_tfidf).to_csv('y_pred_xgb_tfidf.csv', index=False)
pd.DataFrame(y_pred_lgb_tfidf).to_csv('y_pred_lgb_tfidf.csv', index=False)
#%%
# NER Kullanımı
# Gerekli kütüphanelerin eklenmesi
import spacy
import en_ner_bc5cdr_md as ner_model
import pandas as pd
#%%
# MTSamples veri setini yükle
mtsample_df = pd.read_csv('mtsamples.csv')

# Boş olmayan transkripsiyonları seç
mtsample_df.dropna(subset=['transcription'], inplace=True)

# NER modelini yükle
nlp = ner_model.load()

# Örnek transkripsiyonları al
sample_transcriptions = mtsample_df['transcription'].sample(n=5, random_state=42)

# NER işlemi
for transcription in sample_transcriptions:
    doc = nlp(transcription)
    
    print("\nTranscription Text:")
    print(transcription)
    
    print("\nNamed Entities:")
    print("TEXT\t\tSTART\tEND\tENTITY TYPE")
    for ent in doc.ents:
        print(f"{ent.text}\t\t{ent.start_char}\t{ent.end_char}\t{ent.label_}")
#%%
# Derin Öğrenme Modeli
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(y_train)

model = Sequential()
model.add(Embedding(input_dim=len(vectorizer_tfidf.get_feature_names_out()), output_dim=32, input_length=X_train_tfidf.shape[1]))
model.add(Conv1D(128, 5, activation='relu'))
model.add(LSTM(64, return_sequences=False))  # Changed here
model.add(Dense(len(label_encoder.classes_), activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train_tfidf, encoded_labels, epochs=10, batch_size=32, validation_split=0.2)
y_pred_dnn_tfidf = model.predict_classes(X_test_tfidf)
#%%
model.save('my_model.h5')

#%%
# Performans Metrikleri
def evaluate_model(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    accuracy = accuracy_score(y_true, y_pred)
    classification_rep = classification_report(y_true, y_pred, labels=classes)
    
    print("Confusion Matrix:")
    print(cm)
    print("\nAccuracy:", accuracy)
    print("\nClassification Report:")
    print(classification_rep)
#%%
print(evaluate_model)