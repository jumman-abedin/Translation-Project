from django import forms
from django.contrib.auth import authenticate
from django.core.validators import RegexValidator

from .helpers import translate_text_using_translator
from .models import User, Translation


class TranslationForm(forms.ModelForm):
    class Meta:
        """Form options."""

        model = Translation
        fields = ['input_language', 'text', 'translated_text']
        widgets = {
            'text': forms.Textarea(),
            'translated_text': forms.Textarea()
        }

    def __init__(self, obj=None, *args, **kwargs):
        super(TranslationForm, self).__init__(*args, **kwargs)
        if obj is not None:
            self.fields['input_language'].initial = obj.input_language
            self.fields['text'].initial = obj.text
            self.fields['translated_text'].initial = obj.translated_text
            self.fields['translated_text'].widget.attrs['disabled'] = True

    def save(self, commit=True):
        instance = super(TranslationForm, self).save(commit=False)
        translationFromUsingTranslator = translate_text_using_translator(instance.input_language, instance.text)
        # translationFromUsingTokenizerAndModel = translate_text_using_tokenizer_and_model(instance.input_language, instance.text)
        # TODO: Use one of the variables from above and assign that variable below to instance.translated_text to save it in the DB.
        instance.translated_text = translationFromUsingTranslator  # or translationFromUsingTokenizerAndModel
        instance.save()
        self.save_m2m()
        return instance


class SignUpForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ['first_name', 'last_name', 'username', 'email', 'bio']
        widgets = {'bio': forms.Textarea()}

    new_password = forms.CharField(
        label='Password',
        widget=forms.PasswordInput(),
        validators=[RegexValidator(
            regex=r'^(?=.*[A-Z])(?=.*[a-z])(?=.*[0-9]).*$',
            message='Password must contain an uppercase letter, a lowercase letter and a number'
        )]
    )
    password_confirmation = forms.CharField(label='Password confirmation', widget=forms.PasswordInput())

    def clean(self):
        super().clean()
        new_password = self.cleaned_data.get('new_password')
        password_confirmation = self.cleaned_data.get('password_confirmation')
        if new_password != password_confirmation:
            self.add_error('password_confirmation', 'Confirmation does not match password.')

    def save(self):
        """Create a new user."""

        super().save(commit=False)
        user = User.objects.create_user(
            self.cleaned_data.get('username'),
            first_name=self.cleaned_data.get('first_name'),
            last_name=self.cleaned_data.get('last_name'),
            email=self.cleaned_data.get('email'),
            bio=self.cleaned_data.get('bio'),
            password=self.cleaned_data.get('new_password'),
        )
        return user


class LogInForm(forms.Form):
    username = forms.CharField(label="Username")
    password = forms.CharField(label="Password", widget=forms.PasswordInput())

    def get_user(self):
        """Returns authenticated user if possible."""

        user = None
        if self.is_valid():
            username = self.cleaned_data.get('username')
            password = self.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
        return user


class UserForm(forms.ModelForm):
    """Form to update user profiles."""

    class Meta:
        """Form options."""

        model = User
        fields = ['first_name', 'last_name', 'username', 'email', 'bio']
        widgets = {'bio': forms.Textarea()}


class PasswordForm(forms.Form):
    """Form enabling users to change their password."""

    password = forms.CharField(label='Current password', widget=forms.PasswordInput())
    new_password = forms.CharField(
        label='Password',
        widget=forms.PasswordInput(),
        validators=[RegexValidator(
            regex=r'^(?=.*[A-Z])(?=.*[a-z])(?=.*[0-9]).*$',
            message='Password must contain an uppercase character, a lowercase '
                    'character and a number'
        )]
    )
    password_confirmation = forms.CharField(label='Password confirmation', widget=forms.PasswordInput())
