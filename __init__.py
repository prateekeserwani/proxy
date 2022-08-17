# to import all python files in the directory

from measures import *
from measures.snip import compute_snip_per_weight
from measures.synflow import compute_synflow_per_weight
from measures.fisher import compute_fisher_per_weight
from measures.prospr import pscore 
from measures.jacob_cov import compute_jacob_cov
import torch.nn.functional as F
import torch 
from utils import *
import numpy as np


def no_op(self,x):
	return x

def copynet(self, bn):
	net = copy.deepcopy(self)
	if bn==False:
		for l in net.modules():
			if isinstance(l,nn.BatchNorm2d) or isinstance(l,nn.BatchNorm1d) :
				l.forward = types.MethodType(no_op, l)
	return net


def find_measure(net_orig,                  # neural network
                  inputs, targets,                # a data loader (typically for training data)
                  dataload_info,             # a tuple with (dataload_type = {random, grasp}, 					number_of_batches_for_random_or_images_per_class_for_grasp, number of classes)
                  device,                    # GPU/CPU device used
                  loss_fn=F.cross_entropy,   # loss function to use within the zero-cost metrics
                  metric='jacob_cov',        # an array of measure names to compute, if left blank, all measures are computed by default
                  measures_arr=None):        # [not used] if the measures are already computed but need to be summarized, pass them here

    #Given a neural net
    #and some information about the input data (dataloader)
    #and loss function (loss_fn)
    #this function returns an array of zero-cost proxy metrics.

	def sum_arr(arr):
		histo=[]

		import matplotlib.pyplot as plt
		from scipy.stats import entropy
		#print('computed score', len(arr))
		#input('halt')
		sum = 0.
		for i in range(len(arr)):
			'''
			print('instant ', str(i),' ',arr[i].shape)
			for j in range(arr[i].shape[1]):
				print(arr[i].shape[2],arr[i].shape[3])
				if arr[j].shape[2]*arr[j].shape[3]>1:
					print(arr[i][1,j,:,:].shape, torch.sum(arr[i][0,j,...]))
					if torch.sum(arr[i][1,j,...])>0:
						plt.imshow(arr[i][1,j,:,:].cpu())
						plt.show()
			'''
			#print(arr[i].shape)
			#input('halt')
			for ii in range(arr[i].shape[0]):
				temp = arr[i][ii,...]
				#print(torch.abs(temp))
				temp = temp.reshape(-1).cpu().numpy() 
				from scipy.stats import kurtosis, skew

				#print(np.sum(temp), kurtosis(temp), skew(temp))

				plt.hist(temp,7)				
				#plt.show()
				plt.savefig('./tt/'+str(i)+'_'+str(ii)+'.jpg')
			histo = histo+arr[i].reshape(-1).tolist() 
			sum += torch.sum(torch.abs(arr[i]))
			plt.clf()

		x=np.asarray(histo) 
		#x = entropy(x)
		return sum.item()

	dataload, num_imgs_or_batches, num_classes = dataload_info

	if not hasattr(net_orig,'get_prunable_copy'):
		net_orig.get_prunable_copy = types.MethodType(copynet, net_orig)

	#move to cpu to free up mem
	torch.cuda.empty_cache()
	#net_orig = net_orig.cpu() 
	#torch.cuda.empty_cache()
	
	done, ds = False, 1
	

	#if measure_names=='snip':
	#try:
	#print('reached here')
	if metric=='snip':
		measure_values = compute_snip_per_weight(net_orig, inputs, targets, loss_fn=loss_fn, split_data=ds)
	elif metric=='synflow':
		measure_values = compute_synflow_per_weight(net_orig, inputs, targets, loss_fn=loss_fn, split_data=ds)
	elif metric=='fisher':
		measure_values = compute_fisher_per_weight(net_orig, inputs, targets, loss_fn=loss_fn, split_data=ds)
	elif metric=='jacob_cov':
		measure_values = compute_jacob_cov(net_orig, inputs, targets, loss_fn=loss_fn, split_data=ds)
	done = True
	'''
	except RuntimeError as e:
		if 'out of memory' in str(e):
			done=False
			if ds == inputs.shape[0]//2:
				raise ValueError(f'Can\'t split data anymore, but still unable to run. Something is wrong') 
			ds += 1
			while inputs.shape[0] % ds != 0:
				ds += 1
			torch.cuda.empty_cache()
			print(f'Caught CUDA OOM, retrying with data split into {ds} parts')
		else:
			raise e
	'''

	measure_values = sum_arr(measure_values)
	#print(measure_values)
	net_orig = net_orig.to(device).train()
	return measure_values
	

#def ps(net, device):
#	pscore(net,device)
		
		
	

