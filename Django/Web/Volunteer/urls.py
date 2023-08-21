"""Volunteer URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from user_manage import views

urlpatterns = [
    # 访问这个url，就执行后面的函数
    path('index', views.index),

    path('index_000615', views.index_000615),
    path('000615', views.index_mystock_000615),
    path('cmp_000615', views.index_cmp_000615),
    path('eval_all_000615', views.eval_all_000615),
    path('eval_000615', views.eval_000615),
    path('time_000615', views.eval_time_000615),

    path('index_000628', views.index_000628),
    path('000628', views.index_mystock_000628),
    path('cmp_000628', views.index_cmp_000628),
    path('eval_all_000628', views.eval_all_000628),
    path('eval_000628', views.eval_000628),
    path('time_000628', views.eval_time_000628),

    path('index_000629', views.index_000629),
    path('000629', views.index_mystock_000629),
    path('cmp_000629', views.index_cmp_000629),
    path('eval_all_000629', views.eval_all_000629),
    path('eval_000629', views.eval_000629),
    path('time_000629', views.eval_time_000629),

    path('index_000635', views.index_000635),
    path('000635', views.index_mystock_000635),
    path('cmp_000635', views.index_cmp_000635),
    path('eval_all_000635', views.eval_all_000635),
    path('eval_000635', views.eval_000635),
    path('time_000635', views.eval_time_000635),

    path('index_000659', views.index_000659),
    path('000659', views.index_mystock_000659),
    path('cmp_000659', views.index_cmp_000659),
    path('eval_all_000659', views.eval_all_000659),
    path('eval_000659', views.eval_000659),
    path('time_000659', views.eval_time_000659),

    path('index_000663', views.index_000663),
    path('000663', views.index_mystock_000663),
    path('cmp_000663', views.index_cmp_000663),
    path('eval_all_000663', views.eval_all_000663),
    path('eval_000663', views.eval_000663),
    path('time_000663', views.eval_time_000663),

    path('index_000665', views.index_000665),
    path('000665', views.index_mystock_000665),
    path('cmp_000665', views.index_cmp_000665),
    path('eval_all_000665', views.eval_all_000665),
    path('eval_000665', views.eval_000665),
    path('time_000665', views.eval_time_000665),

    path('index_000666', views.index_000666),
    path('000666', views.index_mystock_000666),
    path('cmp_000666', views.index_cmp_000666),
    path('eval_all_000666', views.eval_all_000666),
    path('eval_000666', views.eval_000666),
    path('time_000666', views.eval_time_000666),

    path('index_000670', views.index_000670),
    path('000670', views.index_mystock_000670),
    path('cmp_000670', views.index_cmp_000670),
    path('eval_all_000670', views.eval_all_000670),
    path('eval_000670', views.eval_000670),
    path('time_000670', views.eval_time_000670),

    path('index_000679', views.index_000679),
    path('000679', views.index_mystock_000679),
    path('cmp_000679', views.index_cmp_000679),
    path('eval_all_000679', views.eval_all_000679),
    path('eval_000679', views.eval_000679),
    path('time_000679', views.eval_time_000679),

    path('index_000680', views.index_000680),
    path('000680', views.index_mystock_000680),
    path('cmp_000680', views.index_cmp_000680),
    path('eval_all_000680', views.eval_all_000680),
    path('eval_000680', views.eval_000680),
    path('time_000680', views.eval_time_000680),

    # path('index', views.index),
    # path('cmp.html', views.index_cmp),
    # path('eval_all.html', views.eval_all),
    # path('time', views.eval_time),
    # path('eval', views.eval)

]
