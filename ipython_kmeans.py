#!/usr/bin/env python

import sys
import os
import math

from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors as rdmd

from sklearn.cluster import MiniBatchKMeans,KMeans

import pandas as pd
from tqdm import tqdm
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from rdkit.Chem import Draw

from scipy.spatial.distance import cdist

from docopt import docopt

fp_bits=int(2048)
fp_dim = fp_bits
num_clusters = int(100) #root(10,000)

infile_name = "antimalarial.smi"
outfile_name = "output_antim.csv"

fp_type = "morgan2"

name_list_fp = ["%s_%d" % (fp_type, i) for i in range(0, fp_bits)] #naming the bit vector columns



def get_numpy_fp(mol):
	fp = rdmd.GetMorganFingerprintAsBitVect(mol, 2, nBits=fp_bits)
	arr = np.zeros((1,), int)
	DataStructs.ConvertToNumpyArray(fp, arr)
	return arr

def get_tanimoto_similarity(mol1,mol2):
	fp1 = rdmd.GetMorganFingerprintAsBitVect(mol1, 2, nBits=fp_bits)
	fp2 = rdmd.GetMorganFingerprintAsBitVect(mol2, 2, nBits=fp_bits)
	tani_Score = DataStructs.TanimotoSimilarity(fp1,fp2)
	return tani_Score

#suppl = Chem.SmilesMolSupplier(infile_name, titleLine=False)



def write_fingerprint_df(df, outfile_name):
	start = time.time()
	df.to_parquet(outfile_name, engine="fastparquet", compression="gzip")
	elapsed = time.time() - start
	print(f"{elapsed:.1f} sec required to write {outfile_name}")


def read_fingerprint_df(fp_file_name):
	start = time.time()
	df = pd.read_parquet(fp_file_name, engine='fastparquet')
	num_rows, num_cols = df.shape
	elapsed = time.time() - start
	print(f"Read {num_rows} rows from {fp_file_name} in {elapsed:.1f} sec, fingerprint dimension is {num_cols - 2}")
	return df


def find_cluster_centers(df,centers):
	center_set = set()
	for k,v in df.groupby("Cluster"):
		fp_list = v.values[0::,3::] # [start:end:skip_by]
		XA = np.array([centers[k]]).astype(float)
		XB = np.array(fp_list).astype(float)
		dist_list = cdist(XA, XB)
		min_idx = np.argmin([dist_list])
		center_set.add(v.Name.values[min_idx])
	return ["Yes" if x in center_set else "No" for x in df.Name.values]

def generate_fingerprint_df(infile_name, fp_type="morgan2", fp_bits=2048):

	suppl = Chem.SmilesMolSupplier(infile_name, titleLine=False)
	fp_list = []
	name_list = []
	smiles_list = []
	print(f"Generating {fp_type} fingerprints with {fp_bits} bits")
	for mol in tqdm(suppl):
		if mol:
			smiles = Chem.MolToSmiles(mol)
			fp_list.append(get_numpy_fp(mol))
			name_list.append(mol.GetProp("_Name"))
			smiles_list.append(smiles)
	start = time.time()
	df = pd.DataFrame(np.array(fp_list), columns=name_list_fp)
	elapsed = time.time() - start
	df.insert(0, "SMILES", smiles_list)
	df.insert(1, "Name", name_list)
	print(f"{elapsed:.1f} sec required to generate dataframe")
	return df

fp_df = generate_fingerprint_df(infile_name, fp_type=fp_type, fp_bits=fp_dim)


train_df = fp_df
arr = np.array(train_df.values[0::, 2::], dtype=np.float16)

#we are taking till k=200 becasue python showed warning that k=68 has unique possible clusters
wcss_kmeans = [] #this list will contain wcss for k=1 to 200
for i in range(201,250):
	print('Doing for k= %d'%i)
	model = KMeans(n_clusters=i) #init Kmeans model with k=i
	model.fit(arr) #model is given data as input
	wcss_kmeans.append(model.inertia_) #model.inertia_ calc wcss and the value is added to wcss list
print(wcss_kmeans)

#we will save wcss values to txt file
with open('wcss_200_values.txt','w') as f:
	for item in wcss_kmeans:
		f.write(str(item).join(' ,'))

#By WCSS vs k graph, we choose k=50



len(wcss_kmeans) #200
y = wcss_kmeans
x = np.arange(1,len(y)+1) #len(x)=200

#plot wcss vs k=1 to 200 but with y axis value limited to max of wcss_kmeans 
plt.clf()
plt.title('WCSS vs k')
plt.xlabel("k")
plt.ylabel("wcss_score")
plt.xlim([0,len(x)])
plt.ylim([0,np.max(wcss_kmeans)])
plt.plot(x,y)
plt.savefig('wcss250.png')

#plot wcss vs k=1 to 200 but with y axis value limited to 1000
plt.clf()
plt.title('WCSS vs k')
plt.xlabel("k")
plt.ylabel("wcss_score")
plt.xlim([0,len(x)])
plt.ylim([9000,20000])
plt.yticks(np.arange(np.min(y), 20000, 1000))
plt.plot(x,y)
plt.savefig('wcss200_1zoom.png')


model = KMeans(n_clusters=50)
model.fit(arr)

df_out = train_df
chunk_size = 500
all_data = np.array(df_out.values[0::, 2::], dtype=bool)
chunks = math.ceil(all_data.shape[0] / chunk_size)
cluster_id_list = []

start = time.time()
for row, names in tqdm(zip(np.array_split(all_data, chunks), np.array_split(df_out[['SMILES', 'Name']].values, chunks)),
					   total=chunks, desc="Processing chunk"):
	p = model.predict(row)
	cluster_id_list += list(p)
elapsed = time.time() - start

df_out.insert(2,"Cluster",cluster_id_list)
center_list = find_cluster_centers(df_out,model.cluster_centers_)
df_out.insert(3,"Center",center_list)
output_df = df_out[["SMILES", "Name", "Cluster","Center"]]
# print(f"Clustered {num_rows} into {num_clusters} in {elapsed:.1f} sec")
output_df.to_csv('out50_df.csv', index=False)


 h = output_df['Cluster'].hist(bins=50)
 fig = h.get_figure()
 fig.savefig('hist.png')

snsdf = output_df
snsdf['Target'] = [x.split("_")[0] for x in snsdf.Name]
snsdf.sort_values("Cluster",inplace=True)
unique_targets = snsdf.Target.unique()
# array(['B-raf', 'Acetylcholinesterase', 'A2a', 'COX-1', 'COX-2',
#        'Antimalarial', 'Caspase', 'Cannabinoid', 'ABL1', 'Aurora-A',
#        'Carbonic'], dtype=object)

target_dict = dict([(x,y) for x,y in zip(unique_targets,range(0,len(unique_targets)))])
num_targets = len(unique_targets)
target_dict
# {'B-raf': 0,
#  'Acetylcholinesterase': 1,
#  'A2a': 2,
#  'COX-1': 3,
#  'COX-2': 4,
#  'Antimalarial': 5,
#  'Caspase': 6,
#  'Cannabinoid': 7,
#  'ABL1': 8,
#  'Aurora-A': 9,
#  'Carbonic': 10}


mat = []
for k,v in snsdf.groupby("Cluster"):
    row_vec = np.zeros(num_targets,dtype=np.int8)
    for tgt,count in v.Target.value_counts().iteritems():
        tgt_idx = target_dict[tgt]
        row_vec[tgt_idx] = count
    mat.append(row_vec)
mat = np.array(mat)
out_df = pd.DataFrame(mat.transpose())
out_df.insert(0,"Target",unique_targets)
out_df


sns.set(rc={'figure.figsize':(12,12)})
sns.set(font_scale=1.25)
hm_df = pd.DataFrame(np.array(out_df.values[0::,1::],dtype=np.int8))
hm_df.replace(0,np.nan,inplace=True)
ax = sns.heatmap(hm_df,annot=True,yticklabels=unique_targets,linewidths=0.25,cmap="YlGnBu")
settings = ax.set(xlabel='Cluster', ylabel='Target')

#now we will try to plot the points.

X = train_df.iloc[:,4:] #taking all the morgan features (2048)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X)
X_pca = pca.transform(X)
print("original shape:   ", X.shape) #original shape:    (1109, 2048)
print("transformed shape:", X_pca.shape) #transformed shape: (1109, 2)

plt.clf()
fig = plt.figure()

plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.scatter(pyes.iloc[:,2],pyes.iloc[:,3],c='black')
plt.xlabel('component_1')
plt.ylabel('component_2')
plt.colorbar()
fig.savefig('pca501.png')























def kmeans_cluster(df, num_clusters, outfile_name, sample_size=None):
	num_rows, num_cols = df.shape
	if num_rows > 10000:
		if sample_size:
			rows_to_sample = sample_size
		else:
			# number of samples needs to at least equal the number of clusters
			rows_to_sample = max(int(num_rows / 10),num_clusters)
		train_df = df.sample(rows_to_sample)
		print(f"Sampled {rows_to_sample} rows")
	else:
		train_df = df
	arr = np.array(train_df.values[0::, 2::], dtype=np.float16)#all rowsall the 2048 bit columns
	wcss = []
	start = time.time()
	km = MiniBatchKMeans(n_clusters=num_clusters, random_state=0, batch_size=3 * num_clusters)
	km.fit(arr)
	print("km.inertia_ (WCSS) is ====>", km.inertia_)
	wcss.append(km.inertia_)

	df_out = df
	chunk_size = 500
	all_data = np.array(df_out.values[0::, 2::], dtype=bool)
	chunks = math.ceil(all_data.shape[0] / chunk_size)
	out_list = []

	# It looks like the predict method chokes if you send too much data, chunking to 500 seems to work
	cluster_id_list = []
	for row, names in tqdm(zip(np.array_split(all_data, chunks), np.array_split(df_out[['SMILES', 'Name']].values, chunks)),
						   total=chunks, desc="Processing chunk"):
		p = km.predict(row)
		cluster_id_list += list(p)
	elapsed = time.time() - start




#fp_df.head(5)
#mol1 = Chem.MolFromSmiles(fp_df.iloc[1,0])
#out_df = pd.read_csv('test_out.csv')
#out83 =  out_df.loc[out_df['Cluster']==83]


# kmeans_cluster(fp_df, num_clusters, outfile_name, sample_size=1000)


tmol = Chem.MolFromSmiles('Cccccc')
tmol_arr = get_numpy_fp(tmol)
tmol_arr.shape
set(np.nonzero(tmol_arr))
reshaped_tmol_arr = tmol_arr.reshape((1,fp_bits))
pmol = km.predict(reshaped_tmol_arr) #predicts the cluster number 
#pmol.shape = (1,)
output_df[(output_df['Cluster']==pmol[0]) & (output_df['Center']=='Yes')]
#pmol[0] because we want to get int value, else its an array with 1 element


for i in predlist[:5]:
	imol = Chem.MolFromSmiles(i)
	imol_arr = get_numpy_fp(imol)
	reshaped_imol_Arr = imol_arr.reshape((1,fp_bits))
	pmol = model.predict(reshaped_imol_Arr)
	print('SMILE==> ',i,' belongs to:: %d cluster'%pmol[0])

for i in range(1000):
	moli = Chem.MolFromSmiles(output_df.iloc[i,0])
	max = 0.0
	idx = 0
	tscore = get_tanimoto_similarity(tmol,moli)
	if (tscore > max):
		max = tscore
		idx = i
print(tscore,i)


km.cluster_centers_.shape #(100, 2048)
tdf = pd.read_csv("test_out.csv")
tdf.shape #(1000, 4)
tdf.columns #Index(['SMILES', 'Name', 'Cluster', 'Center'], dtype='object')
tdf['Target'] = [x.split("_")[0] for x in tdf.Name]
tdf.columns #Index(['SMILES', 'Name', 'Cluster', 'Center', 'Target'], dtype='object')
tdf.iloc[:1,:]
#                          SMILES     Name  Cluster Center Target
#0  CNCC1CC2c3ccccc3Cc3ccccc3N2O1  A2a_000       83     No    A2a


unique_targets = tdf.Target.unique() #Names of original cluster labels 

unique_targets
#array(['A2a', 'ABL1', 'Acetylcholinesterase', 'Aurora-A', 'B-raf',
#       'COX-1', 'COX-2', 'Cannabinoid', 'Carbonic', 'Caspase'],dtype=object)

target_dict = dict([(x,y) for x,y in zip(unique_targets,range(0,len(unique_targets)))])

target_dict
# {
#  'A2a': 0,
#  'ABL1': 1,
#  'Acetylcholinesterase': 2,
#  'Aurora-A': 3,
#  'B-raf': 4,
#  'COX-1': 5,
#  'COX-2': 6,
#  'Cannabinoid': 7,
#  'Carbonic': 8,
#  'Caspase': 9
# }

num_targets = len(unique_targets)


mat = []
for k,v in df.groupby("Cluster"):#grouping rows by same cluster
	row_vec = np.zeros(num_targets,dtype=np.int8)#initialize row_vector
	for tgt,count in v.Target.value_counts().iteritems(): #getting count of each 'Name' within cluster
		tgt_idx = target_dict[tgt] #getting index of 'Name' from target_dict. For ex:- 'A2a': 0
		row_vec[tgt_idx] = count #putting the value of count of A2a in 0th index of row_vector
	mat.append(row_vec) #adding all such 100 row_vectors 
mat = np.array(mat)
mat.shape #(100, 10) 100rows for 100 clusters, 10cols for 10 'Name'
out_tdf = pd.DataFrame(mat.transpose()) #Transposing the matrix ==> out_tdf.shape (10, 100)
out_tdf.insert(0,"Target",unique_targets)#adding 'Target' col at start (0th index) with values of 'Name'
out_tdf #to print out_df

#The matrix above isn't a great visualization, let's make a heatmap.

sns.set(rc={'figure.figsize':(12,12)})
sns.set(font_scale=1.25)
hm_df = pd.DataFrame(np.array(out_tdf.values[0::,1::],dtype=np.int8))
hm_df.replace(0,np.nan,inplace=True)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X)
X_pca = pca.transform(X)
print("original shape:   ", X.shape)
print("transformed shape:", X_pca.shape)



fig = plt.figure()

plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar();

fig.savefig('pca.png')

plot_df = train_df.iloc[:,:4] #ignoring morgan_2048 columns
plot_df["x_val"] = X_pca[:,0] #adding x-coord column calc. from PCA
plot_df["y_val"] = X_pca[:,1] #adding y-coord column calc. from PCA

#Now we can plot the points



model = KMeans(n_clusters=100)
model.fit(arr)
yhat = model.predict(X)
clusters = np.unique(yhat)
kmeans_df = train_df.iloc[:,:4]
kmeans_df['Kmeans_Cluster'] = yhat #added cluster numbers made using simple Kmeans 
kmeans_df.to_csv('kmeans_df.csv')



# wcss_array = np.array(wcss_kmeans) #converting list to numpy array



