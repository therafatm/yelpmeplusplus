import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth

with open('100kbusiness.pickle','rb') as fp:
	data = pickle.load(fp)

print ('Formatting Data')
formated_data = []

count = 0
for i in range(len(data['stars'])):
	if data['city'][i] == 'phoenix' or data['city'][i] == 'Phoenix': 
		if(count < 200):
			formated_data.append((data['latitude'][i], data['longitude'][i], round(data['stars'][i])))
			count += 1

formated_data = np.asarray(formated_data)

print ('Estimating Bandwidth')
bandwidth = estimate_bandwidth(formated_data, n_jobs=-1)

ms = MeanShift(bandwidth=bandwidth, cluster_all=True, n_jobs=-1)

print('Performing Clustering')
results = ms.fit_predict(formated_data)

formated_data_list = formated_data.tolist()
results = results.tolist()
centroids = ms.cluster_centers_.tolist()

print('Number of clusters: ' + str(len(centroids)))


for li,elm in zip(formated_data_list,results):
	li[2] = int(li[2])
	li.append(elm)
	
with open('data.js','w') as outfile:
	outfile.write('var locations = ')
	json.dump(formated_data_list, outfile)
	outfile.write(';')
	outfile.write('var centroids = ')
	json.dump(centroids, outfile)
	outfile.write(';')

data = (formated_data,results,centroids)
with open('outData.p','wb') as outfile:
	pickle.dump(data,outfile,protocol = 4)
