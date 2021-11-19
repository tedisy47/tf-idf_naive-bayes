# from django.shortcuts import render
from django.shortcuts import  render, redirect, HttpResponse


# from django.contrib.auth.models import User
# from django.contrib.auth import authenticate, login as auth_login, logout as auth_logout
# from django.contrib import messages

from django.core.files.storage import FileSystemStorage

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

import pandas as pd
import numpy as np
import math

import tfidf.naive_bayes as nb

import tfidf.tfidf as tf

import tfidf.test as ts
# import requests
import json

# Create your views here.
def index(request):
	if request.method == 'POST':
		# print('test')
		for f in request.FILES.getlist('dataset'):
			filename = f.name
		fs = FileSystemStorage()
		filename = fs.save(filename, f)
		uploaded_file_url = fs.url(filename)
		tf.tokenize(filename)
		return redirect('proses')
	else:
		context	= {
			'page_title': 'sentimen detector',
			'title': 'sentimen detector',
			'dataset' : request.user

		}

		return render(request,'form_dataset.html', context)

# def register(request):
# 	return render(request,'register.html')
def proses(request):
	df = pd.read_csv('hasil.csv',index_col=[0])
	hasil = df.to_html(classes=["table", "table-bordered", "table-striped", "table-hover"])
	context	= {
		'page_title': 'sentimen detector',
		'title': 'proses Term-Frequecy',
		'dataset' : request.user,
		'hasil' : hasil

	}

	return render(request,'proses.html', context)

def perior(request):	
	df = pd.read_csv('hasil.csv',index_col=[0])
	perior = tf.perior(df)
	hasil = perior.to_html(classes=["table", "table-bordered", "table-striped", "table-hover"])
	context = {
		'page_title': 'sentimen detector',
		'title': 'proses perior',
		'hasil' : hasil
	}
	return render(request,'proses2.html',context)


def conditional(request):
	df = pd.read_csv('hasil.csv',index_col=[0])
	conditional = tf.conditional(df)
	hasil = conditional.to_html(classes=["table", "table-bordered", "table-striped", "table-hover"])
	context = {
		'page_title': 'sentimen detector',
		'title': 'proses conditional',
		'hasil' : hasil
	}
	return render(request,'proses2.html',context)


def dataset(request):
	df = pd.read_csv('data_set.csv',index_col=[0])
	hasil = df.to_html(classes=["table", "table-bordered", "table-striped", "table-hover"])
	context = {
		'page_title': 'sentimen detector',
		'title': 'hasil dataset',
		'hasil' : hasil
	}
	return render(request,'proses2.html',context)

# teting
# teting
# teting
# teting
# teting
# teting
# teting
# teting
# teting
# teting
# teting
# teting






def test_insert(request):
	if request.method == 'POST':
		# print('test')
		post = request.POST
		komentar = post['komentar']
		test = ts.init(komentar)
		df = pd.read_csv('test_tf.csv',index_col=[0])
		hasil = df.to_html(classes=["table", "table-bordered", "table-striped", "table-hover"])
		context	= {
			'page_title': 'sentimen detector',
			'title': 'proses Term-Frequecy',
			'dataset' : request.user,
			'hasil' : hasil

		}

		return render(request,'proses_test.html', context)
	else:
		context	= {
			'page_title': 'sentimen detector',
			'title': 'sentimen detector',
			'dataset' : request.user

		}

		return render(request,'form_test.html', context)

def matching(request):
	df = pd.read_csv('test_tf.csv',index_col=[0])
	with open('token.json') as f_in:
		token = json.load(f_in)
	matching = ts.matching(df,token)
	hasil = matching.to_html(classes=["table", "table-bordered", "table-striped", "table-hover"])
	context = {
		'page_title': 'sentimen detector',
		'title': 'Matching',
		'hasil' : hasil,
	}
	return render(request,'proses2.html',context)
	
def hitung_cp(request):
	df = pd.read_csv('matching.csv',index_col=[0])
	
	hitung_cp = ts.hitung_cp(df)
	
	p = hitung_cp['p']
	p = pd.DataFrame(p, [1])
	hasil = p.to_html(classes=["table", "table-bordered", "table-striped", "table-hover"])
	

	data = hitung_cp['data']
	data = pd.DataFrame(data, [1])
	hasil_data = data.to_html(classes=["table", "table-bordered", "table-striped", "table-hover"])

	context = {
		'page_title': 'sentimen detector',
		'title': 'hasil perior',
		'hasil' : hasil,
		'data' : hasil_data,
		'hasil_klasifikasi' : hitung_cp['hasil'],
		'nilai_tertinggi' : hitung_cp['nilai_tertinggi'] 
	}
	return render(request,'hasil_test.html',context)
	