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
import matplotlib.pyplot as plt


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
no_of_arch=30


api = API(model_space, verbose=False)

per_iteration_record=[]
all_nets=[]
all_optimizer=[]

def load_model():
	for i, arch_str in enumerate(api):
		if i>=no_of_arch:
			break
		net = nasbench2.get_model_from_arch_str(arch_str, number_of_classes)
		net.to('cpu')
		all_optimizer.append(torch.optim.Adam(net.parameters(), lr=0.001))
		all_nets.append(net)

def update_model(net, dataloader,optimizer,iteration, loss_fn=F.cross_entropy):
	ll=0
	for j, data in enumerate(tqdm(dataloader)):
		inputx, labels = data
		optimizer.zero_grad()
		outputs = net(inputx.to(device))
		loss = loss_fn(outputs, labels.to(device))
		loss.backward()
		optimizer.step()
		ll=ll+loss.item()
	print('loss', ll/j)
	
def training_for_epochs(arch_no):
	train_loader, val_loader = get_cifar_dataloaders(batch_size, batch_size, dataset, num_data_workers)
	#inputs, targets = prepare_proxy_dataset(dataset, device=device)
	#train_loader_2, val_loader_2 = get_cifar_dataloaders(batch_size, batch_size, dataset, num_data_workers)
	load_model()
	net = all_nets[arch_no]
	optimizer= all_optimizer[arch_no]
	net.to(device)
	net.train()
	
	for i in range(number_of_epoch):
		print('EPOCH => ', i, '['+str(arch_no)+']')
		update_model(net, train_loader, optimizer, i)
		torch.save(net.state_dict(), './analysis_2/'+str(arch_no)+'_'+str(i)+'.pth')

def trainig():
	for i in range(no_of_arch):
		training_for_epochs(i)

def compute_accuracy(dataloader, net):
	output=[]
	for j, data in enumerate(tqdm(dataloader)):
		inputx, labels = data
		labels=labels.to(device)
		outputs = net(inputx.to(device))
		outputs = F.softmax(outputs)
		outputs = outputs.argmax(dim=1)
		output=output+ (outputs==labels).cpu().tolist()
	output = np.asarray(output, dtype=np.uint8)
	accuracy= np.sum(output)/output.shape[0]
	return accuracy 

def model_evaluation(train_loader, val_loader,net):
	print('Training=>')	
	train_accuracy = compute_accuracy(train_loader, net)
	print(train_accuracy)		
	print('Validation=>')
	val_accuracy= compute_accuracy(val_loader, net)	
	print(val_accuracy)
	return train_accuracy, val_accuracy

########################
def per_epoch_evalutation():
	load_model()
	train_loader, val_loader = get_cifar_dataloaders(batch_size, batch_size, dataset, num_data_workers)
	inputs, targets = get_some_data(train_loader, num_batches=dataload_info, device=device)
	#inputs, targets = prepare_proxy_dataset(dataset, device=device)
	epoch_train=[]
	epoch_val=[]
	for i in range(no_of_arch):
		train=[]
		val=[]
		for j in range(number_of_epoch):
			print('architecture =>',i, 'epoch =>', j+1)
			print(api[i])
			net = nasbench2.get_model_from_arch_str(api[i], number_of_classes)
			PATH='./analysis_2/'+str(i)+'_'+str(j)+'.pth'
			net.load_state_dict(torch.load(PATH))
			net.cuda()
			print('weight loaded')
			net.eval()
			train_accuracy, val_accuracy= model_evaluation(train_loader, val_loader,net)
			train.append(train_accuracy)
			val.append(val_accuracy)
		train = np.asarray(train)
		val= np.asarray(val)
		epoch_train.append(train)
		epoch_val.append(val)
	return epoch_train, epoch_val

############################

def epoch_vs_snip(viz=False):
	load_model()
	train_loader, val_loader = get_cifar_dataloaders(batch_size, batch_size, dataset, num_data_workers)
	inputs, targets = get_some_data(train_loader, num_batches=dataload_info, device=device)
	#inputs, targets = prepare_proxy_dataset(dataset, device=device)
	epoch_measure=[]
	for i in range(no_of_arch):
		measure=[]
		for j in range(number_of_epoch):
			print('architecture =>',i, 'epoch =>', j+1)
			print(api[i])
			net = nasbench2.get_model_from_arch_str(api[i], number_of_classes)
			PATH='./analysis_2/'+str(i)+'_'+str(j)+'.pth'
			net.load_state_dict(torch.load(PATH))
			net.cuda()
			print('weight loaded')
			net.eval()
			compute_measures = find_measure(net,inputs, targets,(dataload, dataload_info, number_of_classes),device, metric='snip')	
			measure.append(compute_measures)
			print(compute_measures)
		measure= np.asarray(measure)
		epoch_measure.append(measure)

		if viz:
			plt.subplot(3,1,1)
			plt.plot(measure, color = 'r')
			plt.subplot(3,1,2)
			plt.plot(train, color = 'g')
			plt.subplot(3,1,3)
			plt.plot(val, color = 'b')
			#plt.show()
			plt.savefig(str(i)+'.png')
			plt.clf()

	return epoch_measure
	
def pre_trained_accuracy():
	test=[]
	train=[]
	val=[]
	for i in range(no_of_arch):
		print('architecture =>',i)
		print(api[i])
		net = nasbench2.get_model_from_arch_str(api[i], number_of_classes)
		info = api.get_more_info(i, 'cifar10-valid' if dataset=='cifar10' else dataset, iepoch=None, hp='200', is_random=False)
		trainacc = info['train-accuracy']
		valacc   = info['valid-accuracy']
		testacc  = info['test-accuracy']
		test.append(testacc)
		train.append(trainacc)
		val.append(valacc)
		print(trainacc,valacc,testacc )
	return train,val,test

def compute_per_epoch_stats():
	hist_measure = epoch_vs_snip()
	#hist_pretrain = pre_trained_accuracy()
	#np.save('history_measure.npy',hist_measure)
	#np.save('history_pretrain.npy',hist_pretrain)

def compute_spearman(measure, train_stats, pretrain_stats):
	train, val=train_stats
	train_p, val_p, test_p = pretrain_stats
	print(measure, val_p)
	s_cofficient_val = spearmanr(-np.asarray(measure), val_p,nan_policy='omit')
	s_cofficient_test = spearmanr(-np.asarray(measure), test_p,nan_policy='omit')
	print(s_cofficient_val.correlation,s_cofficient_test.correlation )
	
	return s_cofficient_val.correlation, s_cofficient_test.correlation
	
def per_epoch_stats():
	hist_train = np.load('train_val.npy')
	epoch_measure = np.load('history_measure_snip.npy')
	hist_pretrain = np.load('history_pretrain.npy')

	epoch_train, epoch_val = hist_train
	pretrain_train, pretrain_val, pretrain_test = hist_pretrain 
	
	spearman_v =[]	
	spearman_t =[]	
	for j in range(number_of_epoch):
		per_epoch_measure= [epoch_measure[i][j] for i in range(no_of_arch)] 
		per_epoch_train = [hist_train[0][i][j] for i in range(no_of_arch)] 
		per_epoch_val = [hist_train[1][i][j] for i in range(no_of_arch)] 
		spearman = compute_spearman(per_epoch_measure, [per_epoch_train, per_epoch_val],[pretrain_train, pretrain_val, pretrain_test])
		spearman_v.append(spearman[0])
		spearman_t.append(spearman[1])
	plt.plot(np.asarray(spearman_v), color = 'r')
	plt.plot(np.asarray(spearman_t), color = 'b')
	plt.show()

def analyze(arch_no, epoch):
	train_loader, val_loader = get_cifar_dataloaders(batch_size, batch_size, dataset, num_data_workers)
	inputs, targets = get_some_data(train_loader, num_batches=dataload_info, device=device)
	net = nasbench2.get_model_from_arch_str(api[arch_no], number_of_classes)
	PATH='./analysis_2/'+str(arch_no)+'_'+str(epoch)+'.pth'
	net.load_state_dict(torch.load(PATH))
	net.cuda()
	net.eval()
	if not os.path.exists('./tt'):
		os.mkdir('tt')
	compute_measures = find_measure(net,inputs, targets,(dataload, dataload_info, number_of_classes),device, metric='snip')	
	if not os.path.exists('./temp/'+str(arch_no)+'_'+str(epoch)):
		os.mkdir('./temp/'+str(arch_no)+'_'+str(epoch))
		os.system('mv ./tt/*.jpg '+ './temp/'+str(arch_no)+'_'+str(epoch) )

	'''

	for i in range(no_of_arch):
		measure=[]
		for j in range(number_of_epoch):
			print('architecture =>',i, 'epoch =>', j+1)
			print(api[i])
			net = nasbench2.get_model_from_arch_str(api[i], number_of_classes)
			PATH='./analysis_2/'+str(i)+'_'+str(j)+'.pth'
			net.load_state_dict(torch.load(PATH))
			net.cuda()
			print('weight loaded')
			net.eval()
			compute_measures = find_measure(net,inputs, targets,(dataload, dataload_info, number_of_classes),device, metric='snip')	
			measure.append(compute_measures)
			print(compute_measures)
		measure= np.asarray(measure)
		epoch_measure.append(measure)

		if viz:
			plt.subplot(3,1,1)
			plt.plot(measure, color = 'r')
			plt.subplot(3,1,2)
			plt.plot(train, color = 'g')
			plt.subplot(3,1,3)
			plt.plot(val, color = 'b')
			#plt.show()
			plt.savefig(str(i)+'.png')
			plt.clf()

	return epoch_measure
	'''


def main():

	# for the training
	#trainig()

	# store the training, testing, and validation accuracies
	#hist=per_epoch_evalutation()
	#np.save('train_val.npy',hist)

	# compute the zero-cost proxies
	#compute_per_epoch_stats()

	# compute the pretraind stats of the model
	#per_epoch_stats()

	# analyze the stats 
	for i in range(50):
		print(i)
		analyze(10,i)


if __name__=='__main__':
	main()

'''
[81.52      , 90.51333333, 81.84      , 83.88333333, 84.49666667,
87.78      , 88.64666667, 88.77333333, 88.81      , 84.11666667,
10.        , 87.65      , 10.        , 87.83666667, 88.3       ,
56.19333333, 89.90333333, 86.19666667, 87.44666667, 84.35666667,
86.92666667, 89.14      , 85.79333333, 86.15      , 81.62333333,
83.87666667, 86.94333333, 86.55      , 82.7       , 84.46      ] 
'''
# 11 , 1	
	
	






