#! /usr/bin/env python
#================================   IMPORTS   ==================================
import os
import numpy as np
#torch
import torch
from torch import nn
from torch.autograd import Variable

#================================   Costume CAE Module   ==================================
class Predictor(torch.nn.Module):
	def __init__(self):
		#machine configs
		self.use_gpu = torch.cuda.is_available()
		self.default_gpu = 0

		#Predict 6D pose, from patch of image in features space
		super(Predictor, self).__init__()
		self.Hidden_layer = nn.Sequential(
			nn.Linear(256,175),
			nn.Tanh(),
			nn.ReLU(),
			nn.Linear(175,120),
			nn.Tanh(),
			nn.ReLU(),
			nn.Linear(120,80),
			nn.Tanh(),
			nn.ReLU())


		self.Output_layer = nn.Linear(80, 6)

	def forward(self,x):
		x = self.Hidden_layer(x)
		x = self.Output_layer(x)
		return(x)



#=============== auxilar function ====================
	def initialize(self, network_path=[]):
		if not network_path==[]:
			if(os.path.exists(network_path)):
				if(self.use_gpu):
					torch.cuda.set_device(self.default_gpu)
					try:	self.cuda().load_state_dict(torch.load(network_path))
					except:	self.cuda().load_state_dict(torch.load(network_path, map_location=lambda storage, loc: storage))
				else:
					self.load_state_dict(torch.load(network_path, map_location=lambda storage, loc: storage))
			else:
				if(self.use_gpu):	self.cuda()

		else:
			if(self.use_gpu):	self.cuda()


	def save(self, net_name = 'Predictor_Net.pt'):
		torch.save(self.state_dict(),net_name)


	def wrap_input(self, inputs):
		if(self.use_gpu):
			inputs = Variable(torch.Tensor(inputs).float().cuda())
		else:
			inputs = Variable(torch.Tensor(inputs).float())

		return inputs














