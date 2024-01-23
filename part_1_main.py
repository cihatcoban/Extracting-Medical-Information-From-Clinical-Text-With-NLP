#https://www.analyticsvidhya.com/blog/2023/02/extracting-medical-information-from-clinical-text-with-nlp/
#pip install spacy
#pip install pandas
#pip install render
#pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_ner_bc5cdr_md-0.5.3.tar.gz
#%%
import spacy
import en_ner_bc5cdr_md as ner_model
from spacy import displacy
import pandas as pd

#%% 
import os
print(os.getcwd())

mtsample_df = pd.read_csv('mtsamples.csv')

text_df = mtsample_df['transcription']
text_df = text_df.dropna()

text=""

for transcription in text_df:
    text = text + transcription

text = text[0:1000000]

#%% 
nlp = ner_model.load()
doc = nlp(text)
image = displacy.render(doc, jupyter=True, style='ent')

#%%
print("TEXT", "START", "END", "ENTITY TYPE")
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)
#%%
mtsample_df.dropna(subset=['transcription'], inplace=True)
mtsample_df_subset = mtsample_df.sample(n=100, replace=False, random_state=42)
mtsample_df_subset.info()
mtsample_df_subset.head()
#%%
from spacy.matcher import Matcher
pattern = [{'ENT_TYPE':'CHEMICAL'}, {'LIKE_NUM': True}, {'IS_ASCII': True}]
matcher = Matcher(nlp.vocab)
matcher.add("DRUG_DOSE", [pattern])
for transcription in mtsample_df_subset['transcription']:
    doc = nlp(transcription)
    matches = matcher(doc)
    for match_id, start, end in matches:
        string_id = nlp.vocab.strings[match_id]  # get string representation
        span = doc[start:end]  # the matched span adding drugs doses
        print(span.text, start, end, string_id,)
        #Add disease and drugs
        for ent in doc.ents:
            print(ent.text, ent.start_char, ent.end_char, ent.label_)
#%%
# 'text' değişkenini bir metin dosyasına kaydet
with open('transcription_text.txt', 'w', encoding='utf-8') as file:
    file.write(text)

print("Transcription text has been saved to 'transcription_text.txt'")
