from django.conf import settings
from django.shortcuts import redirect
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM


class TranslatorModel:
    def __init__(self, task, model, description):
        self.task = task
        self.model = model
        self.description = description


translators = {
    'EN': TranslatorModel('translation_en_to_es', 'Helsinki-NLP/opus-mt-en-es', 'English to Spanish'),
    'ES': TranslatorModel('translation_es_to_en', 'Helsinki-NLP/opus-mt-es-en', 'Spanish to English')
}


def login_prohibited(view_function):
    def modified_view_function(request):
        if request.user.is_authenticated:
            return redirect(settings.REDIRECT_URL_WHEN_LOGGED_IN)
        else:
            return view_function(request)

    return modified_view_function


def translate_text_using_translator(source_language_key, source_text):
    translatorModel = translators.get(source_language_key)
    translator = pipeline(task=translatorModel.task, model=translatorModel.model)
    return translator(source_text)[0].get('translation_text')


def translate_text_using_tokenizer_and_model(source_language_key, source_text):
    translatorModel = translators.get(source_language_key)
    tokenizer = AutoTokenizer.from_pretrained(translatorModel.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(translatorModel.model)
    input_ids = tokenizer(source_text, return_tensors="pt").input_ids
    outputs = model.generate(input_ids=input_ids, num_beams=5, num_return_sequences=1)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
