# models.py file
from django.core.validators import RegexValidator, MinValueValidator, MaxValueValidator, MaxLengthValidator
from django.db import models
from django.contrib.auth.models import AbstractUser
from libgravatar import Gravatar
from django.conf import settings


class User(AbstractUser):
    """User model used for authentication and microblog authoring."""

    username = models.CharField(
        max_length=30,
        unique=True,
        validators=[RegexValidator(
            regex=r'^@\w{3,}$',
            message='Username must consist of @ followed by at least three alphanumericals'
        )]
    )
    first_name = models.CharField(max_length=50, blank=False)
    last_name = models.CharField(max_length=50, blank=False)
    email = models.EmailField(unique=True, blank=False)
    bio = models.CharField(max_length=520, blank=True)
    followers = models.ManyToManyField(
        'self', symmetrical=False, related_name='followees'
    )

    class Meta:
        """Model options."""

        ordering = ['last_name', 'first_name']

    def full_name(self):
        return f'{self.first_name} {self.last_name}'

    def gravatar(self, size=120):
        """Return a URL to the user's gravatar."""
        gravatar_object = Gravatar(self.email)
        gravatar_url = gravatar_object.get_image(size=size, default='mp')
        return gravatar_url

    def mini_gravatar(self):
        """Return a URL to a miniature version of the user's gravatar."""
        return self.gravatar(size=60)

    def toggle_follow(self, followee):
        """Toggles whether self follows the given followee."""

        if followee == self:
            return
        if self.is_following(followee):
            self._unfollow(followee)
        else:
            self._follow(followee)

    def _follow(self, user):
        user.followers.add(self)

    def _unfollow(self, user):
        user.followers.remove(self)

    def is_following(self, user):
        """Returns whether self follows the given user."""

        return user in self.followees.all()

    def follower_count(self):
        """Returns the number of followers of self."""

        return self.followers.count()

    def followee_count(self):
        """Returns the number of followees of self."""

        return self.followees.count()


class Translation(models.Model):
    author = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    INPUT_LANGUAGE_CHOICES = [
        ("EN", "English"),
        ("ES", "Spanish"),
    ]
    input_language = models.CharField(max_length=280, choices=INPUT_LANGUAGE_CHOICES)
    text = models.CharField(max_length=280)
    translated_text = models.CharField(max_length=280, blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
