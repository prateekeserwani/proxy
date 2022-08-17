import torch
import torch.nn as nn
import pickle
from nas_201_api import NASBench201API as API
from dataset import *
from models import *
from measures import find_measure#, ps
from utils import *
import numpy as np
from scipy.stats import spearmanr
import torch.nn.functional as F
from tqdm import tqdm
from copy import deepcopy

model_space='data/NAS-Bench-201-v1_0-e61699.pth'
seed = 42
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
batch_size = 256
dataset = 'cifar10'
num_data_workers = 8
number_of_classes=10
device=torch.device("cuda:"+str(0) if torch.cuda.is_available() else "cpu")
measures=['snip']
dataload='random'
dataload_info=1
number_of_epoch=50
no_of_arch=10


api = API(model_space, verbose=False)

per_iteration_record=[]
all_nets=[]
all_optimizer=[]

def load_model():
	for i, arch_str in enumerate(api):
		if i>=no_of_arch:
			break
		net = nasbench2.get_model_from_arch_str(arch_str, number_of_classes)
		#print(net)
		#input('halt')
		net.to('cpu')
		#net.to(device)
		all_optimizer.append(torch.optim.Adam(net.parameters(), lr=0.001))
		all_nets.append(net)
		#print(i)

def update_model(dataloader,loss_fn=F.cross_entropy):
	
	for i in range(len(all_nets)):
		netx = all_nets[i]
		net = netx
		#net = net.copy()
		net.to(device)
		net.train()
		optimizer= all_optimizer[i]
		#print('Epoch ==>', str(i+1))
		ll=0
		for j, data in enumerate(tqdm(dataloader)):
			#inputs, targets = get_some_data(dataloader, num_batches=dataload_info, device=device)
			inputx, labels = data
		
			optimizer.zero_grad()
			#print('input size', inputs.shape)
			outputs = net(inputx.to(device))
			loss = loss_fn(outputs, labels.to(device))
			#print('loss function', loss)
			loss.backward()
			optimizer.step()
			ll=ll+loss.item()
		print('loss', ll/j)
		del ll	
		all_nets[i]=net.to('cpu')
		all_optimizer[i]=optimizer

 
def per_epoch_analysis(inputs, targets, iteration):
	measure_array=[]
	
	raw=[]
	counter=0
	
	for i, arch_str in enumerate(api):
		if i>=no_of_arch:
			break
	
		res = {'i':i,'counter':counter, 'arch':arch_str}
		net = all_nets[i].to(device)
		#ps(net,device)
		for metric in measures:
			
			compute_measures = find_measure(net,inputs, targets,(dataload, dataload_info, number_of_classes),device, metric='snip')	
			res['logmeasures']= compute_measures
	
		info = api.get_more_info(i, 'cifar10-valid' if dataset=='cifar10' else dataset, iepoch=None, hp='200', is_random=False)

		trainacc = info['train-accuracy']
		valacc   = info['valid-accuracy']
		testacc  = info['test-accuracy']
		
		res['trainacc']=trainacc
		res['valacc']=valacc
		res['testacc']=testacc
		measure_array.append(res)
		counter=counter+1
		raw.append([i, counter, compute_measures, trainacc, valacc, testacc])
	raw=np.asarray(raw)
	''''
	# sorted according to the computed measure
	#if metric=='snip':
	sorted_matrix = np.flip(raw[raw[:, 2].argsort()],0)
	#elif metric =='synflow':
	#sorted_matrix = raw[raw[:, 2].argsort()]
	sorted_matrix = np.c_[sorted_matrix, np.linspace(1,no_of_arch,no_of_arch).T]


	# sorted according to the testing accuracy
	sorted_matrix = np.flip(sorted_matrix[sorted_matrix[:, 5].argsort()],0)
	sorted_matrix = np.c_[sorted_matrix, np.linspace(1,no_of_arch,no_of_arch).T]

	#print(sorted_matrix)
	#input('halt')
	

	s_cofficient = spearmanr(sorted_matrix[:,-2], sorted_matrix[:,-1])
	'''
	s_cofficient = spearmanr(raw[:,2], raw[:,-1])
	print('iteration', iteration, 's cofficient ', s_cofficient)
	per_iteration_record.append([iteration, s_cofficient])
	

train_loader, val_loader = get_cifar_dataloaders(batch_size, batch_size, dataset, num_data_workers)
#inputs, targets = get_some_data(train_loader, num_batches=dataload_info, device=device)
inputs, targets = prepare_proxy_dataset(dataset, device=device)
train_loader_2, val_loader_2 = get_cifar_dataloaders(batch_size, batch_size, dataset, num_data_workers)


load_model()
print('number of architecture', len(all_nets))
for i in range(number_of_epoch):
	print('EPOCH => ', i)
	per_epoch_analysis(inputs, targets, i)
	update_model(train_loader_2)
per_iteration_record=np.asarray(per_iteration_record)
print('final report')
print(per_iteration_record)

	



		
	
	






