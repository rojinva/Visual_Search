from django.conf.urls import patterns, include, url
from search_manage import views

from django.contrib import admin
admin.autodiscover()

urlpatterns = patterns('',
    # Examples:
    # url(r'^$', 'search_webui.views.home', name='home'),
    # url(r'^blog/', include('blog.urls')),
    url(r'^$', views.index, name='index'),
    url(r'^show_similar$', views.show_similar, name='show_similar'),
    url(r'^show_debug$', views.show_debug, name='show_debug'),
    url(r'^label$', views.label, name='label'),
)
