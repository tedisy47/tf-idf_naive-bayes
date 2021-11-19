import pandas as pd
import numpy as np
import math







def perior(data,key_doc,filename):
	df = pd.read_csv('media/'+filename)
	dokumen = df.index
	jumlah_dokumen = len(dokumen)

	doc = df
	doc = doc.groupby(['sentimen'])
	doc = doc.size().reset_index(name='counts')
	doc['jumlah_dokumen'] = jumlah_dokumen
	doc['perior'] = doc['counts']/ jumlah_dokumen	
	
	return doc

	# hasil = pd.DataFrame(data)
	# print(hasil)

def conditional(data,key_doc):
	
	total_frequesi = 0
	c_frequesi = {}
	for x in key_doc:
		total = data[x].sum()
		total_frequesi += total
		c_frequesi[x]= total
		hasil= []
		for index, y in data.iterrows():
			hasil.append(str(y[x])+'/'+str(total))


		data['P('+str(x)+')'] = hasil

	hasil = {
			'data' : data,
			'total_frequesi' : total_frequesi,
			'c_frequesi' : c_frequesi
	}

	return hasil



def laplace_smoothing(data,key_doc):
	# print(data)
	data_set = data['data']
	c_frequesi = data['c_frequesi']
	hasil_ls = {}
	for x in key_doc:
		# pass
		ls = []
		hasil = []
		for index, y in data_set.iterrows():
			# Laplace smoothing
			ls.append('('+str(y[x])+'+1)/('+str(c_frequesi[x])+'+'+str(data['total_frequesi'])+')')
			penyemut = y[x] + 1
			pembagi = c_frequesi[x] + data['total_frequesi']
			hasil.append(penyemut/pembagi)



		data_set['LS('+str(x)+')'] = ls
		data_set['hasil_'+str(x)] = hasil
		data_set.drop('P('+str(x)+')',axis=1, inplace=True)

	return data_set
	# creat_csv(csv,key_doc)

def create_csv(data,key_doc):
	for x in key_doc:		
		data.drop('LS('+str(x)+')',axis=1, inplace=True)

	data.to_csv('data_set.csv',index=True)
	print(data)