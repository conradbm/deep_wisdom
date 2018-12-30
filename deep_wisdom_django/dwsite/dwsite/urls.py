from django.contrib import admin
from django.urls import include, path

urlpatterns = [
	path('',include('deepwisdom.urls')),
    path('deepwisdom/', include('deepwisdom.urls')),
    path('admin/', admin.site.urls),
]