#! /usr/bin/env python
#================================   IMPORTS   ==================================
import matplotlib.pyplot as plt
import os
import numpy as np
#torch
import torch
from torch import nn
from torch.autograd import Variable

#================================   Costume CAE Module   ==================================
class CAE(torch.nn.Module):
	def __init__(self,Features=256):
		#machine configs
		self.use_gpu = torch.cuda.is_available()
		self.default_gpu = 0

		#Convolutional Network
		super(CAE, self).__init__()
		self.encoder = nn.Sequential(
		    nn.Conv2d(4, 32, 5, stride=1, padding=2),  # b, 16, 10, 10
		    nn.PReLU(),
		    nn.MaxPool2d(2),
		    nn.Conv2d(32, 64, 5, stride=1, padding=2),
		    nn.PReLU()
		    )  # b, 8, 2, 2
		self.fully_connected_encoder = nn.Sequential(
			nn.Linear(64*16*16,1024),
			nn.Linear(1024,Features)
			)

		self.fully_connected_decoder = nn.Sequential(
			nn.Linear(Features,1024),
			nn.Linear(1024,64*16*16)
			)

		self.decoder = nn.Sequential(
			nn.PReLU(),
			nn.ConvTranspose2d(64,64,2,stride=2),
			nn.PReLU(),
		    nn.Conv2d(64, 32, 5, stride=1, padding=2),
			nn.PReLU(),
		    nn.Conv2d(32, 4, 5, stride=1, padding=2),
			nn.PReLU(),
		    )  # b, 8, 2, 2


	def encode(self,x):
		x = self.encoder(x)
		x = x.view(x.size(0), -1)			#resize input from 64*16*16 to 1*16384
		x = self.fully_connected_encoder(x)
		return(x)

	def decode(self,x):
		x = self.fully_connected_decoder(x)
		x = x.view(x.size(0), 64,16,16)	#resize input from 1*16384 to 64*16*16
		x = self.decoder(x)
		return(x)

	def forward(self,x):
		x = self.encode(x)
		x = self.decode(x)		
		return(x)


#=============== auxilar function ====================
	def initialize(self, network_path = []):
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
				

	def wrap_input(self, inputs):
		#reajust input: N * 32' * 32* 4 -> N * 4 * 32' * 32
		inputs = np.swapaxes(inputs,1,3)
		inputs = np.swapaxes(inputs,3,2)	

		#Wrap
		if(self.use_gpu):
			return Variable(torch.Tensor(inputs).cuda())
		else:
			return Variable(torch.Tensor(inputs))

	def unwrap_input(self, inputs):
		#convert Variable dtype to Numpy array
		inputs = inputs.cpu().data.numpy()

		#reajust input: N * 4 * 32' * 32 -> N * 32' * 32* 4
		inputs = np.swapaxes(inputs, 1, 2)
		inputs = np.swapaxes(inputs, 2, 3)

		return inputs

	def wrap_features(self, features):
		#Wrap
		if(self.use_gpu):
			return Variable(torch.Tensor(features).cuda())
		else:
			return Variable(torch.Tensor(features))

	def unwrap_features(self, features):
		return 	features.cpu().data.numpy()



	def save(self, net_name = 'Predictor_Net.pt'):
		torch.save(self.state_dict(),net_name)







#===============================================================
#----------------- Visualize Results ---------------------------
#===============================================================
	def Visualize_Results(self, axes=[], inputs=[], Features=[], outputs=[], rerun=False, show_time=0.25, axis11='off'):
		if rerun:
			Features = self.encode(inputs)
			outputs = self.decode(Features)


		#Unwrap Variables
		inputs = self.unwrap_input(inputs)
		outputs = self.unwrap_input(outputs)
		if len(Features)>0:
			Features = self.unwrap_features(Features)

			for img_in, img_out, feat in zip(inputs, outputs, Features): 
				#Before encryption
				before_rgb = img_in[:,:,0:3]
				before_depth = img_in[:,:,3]
				axes[0,0].set_title('Before encyrption RGB:')
				axes[0,0].axis('off')
				axes[0,0].imshow(before_rgb)
				axes[1,0].set_title('Before encyrption Depth:')
				axes[1,0].axis('off')
				axes[1,0].imshow(before_depth)


				#Encrypted
				encrypted = feat.reshape(16,16)
				axes[0,1].set_title('Encrypted (16x16 representation):')
				axes[0,1].axis('off')
				axes[0,1].imshow(encrypted)
				axes[1,1].axis(axis11)

				#After Decryption
				after_rgb = img_out[:,:,0:3]
				after_depth = img_out[:,:,3]
				axes[0,2].set_title('After encyrption RGB:')
				axes[0,2].axis('off')
				axes[0,2].imshow(after_rgb)
				axes[1,2].set_title('After encyrption Depth:')
				axes[1,2].axis('off')
				axes[1,2].imshow(after_depth)


				plt.pause(show_time)
				axes[0,0].clear()
				axes[1,0].clear()
				axes[0,1].clear()
				axes[0,2].clear()
				axes[1,2].clear()


		else:
			for img_in, img_out in zip(inputs, outputs): 
				#Before encryption
				before_rgb = img_in[:,:,0:3]
				before_depth = img_in[:,:,3]
				axes[0,0].set_title('Before encyrption RGB:')
				axes[0,0].axis('off')
				axes[0,0].imshow(before_rgb)
				axes[1,0].set_title('Before encyrption Depth:')
				axes[1,0].axis('off')
				axes[1,0].imshow(before_depth)

				#Encrypted
				axes[0,1].axis('off')
				axes[1,1].axis(axis11)


				#After Decryption
				after_rgb = img_out[:,:,0:3]
				after_depth = img_out[:,:,3]
				axes[0,2].set_title('After encyrption RGB:')
				axes[0,2].axis('off')
				axes[0,2].imshow(after_rgb)
				axes[1,2].set_title('After encyrption Depth:')
				axes[1,2].axis('off')
				axes[1,2].imshow(after_depth)


				plt.pause(show_time)
				axes[0,0].clear()
				axes[1,0].clear()
				axes[0,2].clear()
				axes[1,2].clear()



















