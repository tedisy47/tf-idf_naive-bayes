from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

import pandas as pd
import numpy as np
import math

import json
# import naive_bayes as nb

import tfidf.tfidf as tf




def testing(data):
	return data
def tfidf(token,kalimat):
	# print(token)
	data = {}
	kata = kalimat.split(' ')
	computeTF = tf.computetf(kata,token)
	
	frequesi = []
	for x in token:
		frequesi.append(computeTF[x])
	
	data['tf'] = frequesi
	df = pd.DataFrame(data,token)
	# print(df)
	df.to_csv('test_tf.csv',index=True)
	with open('token.json', 'w') as f:
		json.dump(token, f)


	return df

	# match = matching(df,token)
	# hitung = hitung_cp(match)
	# print(hitung)



def hitung_cp(match):
	data = {}
	data['positif'] = match['hasil_positif'].sum()
	data['netral'] = match['hasil_netral'].sum()
	data['negatif'] = match['hasil_negatif'].sum()
	# print('test')
	p = {}
	c = 1 / 3 
	p['positif'] = c * data['positif']
	p['netral'] = c * data['netral']
	p['negatif'] = c * data['negatif']

	hasil = max(p, key=p.get)
	# hasil = max(p)
	nilai_tertinggi = p[hasil]
	result = {'p' : p, 'data' : data, 'hasil': hasil,'nilai_tertinggi':nilai_tertinggi}
	# print(result)
	return result


def matching(df,token):
	data_set = pd.read_csv('data_set.csv',index_col=[0])
	netral = []
	negatif = []
	positif = []
	drop = []
	for x in token:
		if x in data_set.index:
			match = data_set.loc[x]
			netral.append(match['hasil_netral'])
			negatif.append(match['hasil_negatif'])
			positif.append(match['hasil_positif'])
		else:
			netral.append(0)
			positif.append(0)
			negatif.append(0)
			drop.append(x)
			# df.drop(x)
	df['positif'] = positif
	df['netral'] = netral
	df['negatif'] = negatif
	df['hasil_positif'] = df['positif'] ** df['tf']
	df['hasil_netral'] = df['netral'] ** df['tf']
	df['hasil_negatif'] = df['negatif'] ** df['tf']
	df.drop(x)

	df.to_csv('matching.csv',index=True)
	return df




def tokenize(kalimat):
	token = []
	kalimat = tf.steming(kalimat)
	kata = kalimat.split(' ')
	for kataToken in kata:
		if kataToken not in token:
			token.append(kataToken)
			# print(kata)

	return token



def init(komentar):
	kalimat = tf.steming(komentar)
	token = tokenize(kalimat)
	tfidif = tfidf(token,kalimat)
	return tfidf


# tokenize(komentar)


# init()