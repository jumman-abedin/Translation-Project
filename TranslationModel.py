from transformers import pipeline
en_es_translator = pipeline("translation_en_to_es", model="Helsinki-NLP/opus-mt-en-es")
res = en_es_translator("My name")
print(res)