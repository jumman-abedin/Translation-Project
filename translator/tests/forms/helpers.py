from django.urls import reverse
from translator.models import Translation


def reverse_with_next(url_name, next_url):
    url = reverse(url_name)
    url += f"?next={next_url}"
    return url


def create_posts(author, from_count, to_count):
    """Create unique numbered translations for testing purposes."""
    for count in range(from_count, to_count):
        text = f'Translation__{count}'
        translation = Translation(author=author, text=text)
        translation.save()


class LogInTester:
    def _is_logged_in(self):
        return '_auth_user_id' in self.client.session.keys()
