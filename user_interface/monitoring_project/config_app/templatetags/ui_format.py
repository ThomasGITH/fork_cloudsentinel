"""CloudSentinel presentation filters."""

from django import template

from config_app.ui_formatting import format_ui_datetime


register = template.Library()


@register.filter(name="ui_datetime")
def ui_datetime(value, fallback="—"):
    return format_ui_datetime(value, fallback)
