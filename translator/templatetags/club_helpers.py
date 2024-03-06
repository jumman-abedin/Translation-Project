from django import template

register = template.Library()


@register.filter(name='myclubstatus')
def myclubstatus(club, user):
    club_member = club.bookclubmembers_set.filter(user=user).first()
    if club_member:
        return club_member.status
    return None
