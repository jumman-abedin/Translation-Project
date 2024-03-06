from django import forms
from django.test import TestCase
from translator.forms import UserForm
from translator.models import User


class EditProfileFormTestCase(TestCase):
    """Unit tests of the user form."""
    fixtures = ['translator/tests/fixtures/default_user.json']

    def setUp(self):
        self.user = User.objects.get(username='@johndoe')

        self.form_input = {
            'username': '@jhondoe2',
            'email': 'johndoe2@example.org',
            'first_name': 'John2',
            'last_name': 'Doe2',
            'bio': 'Bio2',
        }

    def test_form_has_necessary_fields(self):
        # test form for all the necessary fields
        form = UserForm()
        self.assertIn('email', form.fields)
        self.assertIn('first_name', form.fields)
        self.assertIn('last_name', form.fields)
        email_field = form.fields['email']
        self.assertTrue(isinstance(email_field, forms.EmailField))
        self.assertIn('bio', form.fields)

    def test_valid_user_form(self):
        form = UserForm(data=self.form_input)
        self.assertTrue(form.is_valid())

    def test_form_uses_model_validation(self):
        self.form_input['email'] = 'bademailexample.org'
        form = UserForm(data=self.form_input)
        self.assertFalse(form.is_valid())

    def test_form_must_save_correctly(self):
        user = User.objects.get(username='@johndoe')
        form = UserForm(instance=user, data=self.form_input)
        before_count = User.objects.count()
        form.save()
        after_count = User.objects.count()
        self.assertEqual(after_count, before_count)
        self.assertEqual(user.email, 'johndoe2@example.org')
        self.assertEqual(user.first_name, 'John2')
        self.assertEqual(user.last_name, 'Doe2')
        self.assertEqual(user.bio, 'Bio2')
