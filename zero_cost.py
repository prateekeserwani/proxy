import torch
import torch.nn as nn
import pickle
from nas_201_api import NASBench201API as API
from dataset import *
from models import *
from measures import find_measure
from utils import *
import numpy as np
from scipy.stats import spearmanr
import torch.nn.functional as F

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
number_of_iteration=500
no_of_arch=10


api = API(model_space)

per_iteration_record=[]
all_nets=[]
all_optimizer=[]

def load_model():
	#all_nets=[]
	for i, arch_str in enumerate(api):
		if i>=no_of_arch:
			break
		net = nasbench2.get_model_from_arch_str(arch_str, number_of_classes)
		net.to('cpu')
		#net.to(device)
		all_optimizer.append(torch.optim.Adadelta(net.parameters(), lr=1))
		all_nets.append(net)
		#print(i)

def update_model(dataloader,loss_fn=F.cross_entropy):
	inputs, targets = get_some_data(dataloader, num_batches=dataload_info, device=device)
	for i in range(len(all_nets)):
		net = all_nets[i]
		net.to(device)
		net.train()
		optimizer= all_optimizer[i]
		optimizer.zero_grad()
		#print('input size', inputs.shape)
		outputs = net(inputs)
		loss = loss_fn(outputs, targets)
		print('loss function', loss)
		loss.backward()
		optimizer.step()	
		all_nets[i]=net.to('cpu')
		all_optimizer[i]=optimizer

 
def per_iteration_analysis(inputs, targets, iteration):
	measure_array=[]
	
	raw=[]
	counter=0
	
	for i, arch_str in enumerate(api):
		if i>=no_of_arch:
			break
	
		res = {'i':i,'counter':counter, 'arch':arch_str}
		#net = nasbench2.get_model_from_arch_str(arch_str, number_of_classes)
		#print(str(i),'-----',arch_str)
		#net.to(device)
		#init_net(net, args.init_w_type, args.init_b_type)
		#print(net)
		net = all_nets[i].to(device)
		for metric in measures:
			
			compute_measures = find_measure(net,inputs, targets,(dataload, dataload_info, number_of_classes),device)	
			#measure_array.append(compute_measures)
			res['logmeasures']= compute_measures
	
		info = api.get_more_info(i, 'cifar10-valid' if dataset=='cifar10' else dataset, iepoch=None, hp='200', is_random=False)

		trainacc = info['train-accuracy']
		valacc   = info['valid-accuracy']
		testacc  = info['test-accuracy']
		
		res['trainacc']=trainacc
		res['valacc']=valacc
		res['testacc']=testacc
		measure_array.append(res)
		#if compute_measures==0.0:
		#	pass
		#else:
		counter=counter+1
		raw.append([i, counter, compute_measures, trainacc, valacc, testacc])
	#print(measure_array)
	raw=np.asarray(raw)
	#print('the raw link is')
	#print(raw)
	#print(raw.shape)

	# sorted according to the computed measure
	sorted_matrix = np.flip(raw[raw[:, 2].argsort()],0)
	sorted_matrix = np.c_[sorted_matrix, np.linspace(1,no_of_arch,no_of_arch).T]


	# sorted according to the testing accuracy
	sorted_matrix = np.flip(sorted_matrix[sorted_matrix[:, 5].argsort()],0)
	sorted_matrix = np.c_[sorted_matrix, np.linspace(1,no_of_arch,no_of_arch).T]


	s_cofficient = spearmanr(sorted_matrix[:,-2], sorted_matrix[:,-1])
	print('iteration', iteration, 's cofficient ', s_cofficient)
	per_iteration_record.append([iteration, s_cofficient])
	

train_loader, val_loader = get_cifar_dataloaders(batch_size, batch_size, dataset, num_data_workers)
inputs, targets = get_some_data(train_loader, num_batches=dataload_info, device=device)

train_loader_2, val_loader_2 = get_cifar_dataloaders(batch_size, batch_size, dataset, num_data_workers)


load_model()
print('number of architecture', len(all_nets))
for i in range(number_of_iteration):
	print('ITERATION => ', i)
	per_iteration_analysis(inputs, targets, i)
	update_model(train_loader_2)
per_iteration_record=np.asarray(per_iteration_record)
print('final report')
print(per_iteration_record)

	



		
	
	






