#! /usr/bin/env python
#================================   IMPORTS   ==================================
import tables
import random
import time
import numpy as np
import torch

#===============my modules====================
from my_libs import predictor_0 as p0
from my_libs import predictor_1 as p1
from my_libs import predictor_2 as p2
from my_libs import predictor_3 as p3



#===============================================================
#------------------------ Configs ------------------------------
#===============================================================
dataset = 'new_codebook.h5'
dataset = 'codebook_zip.h5'

Predictors = {0: p0.Predictor(), 1: p1.Predictor(), 2: p2.Predictor(), 3: p3.Predictor()}
Name = ['Predictor_0','Predictor_1','Predictor_2','Predictor_3']

epochs = 5

print_stats = 1000
speed_run = False	#Run only 10 prints per stage
speed_run = True	#Run only 10 prints per stage



#===============================================================
#---------------------- Regist Sats ----------------------------
#===============================================================
class stats(tables.IsDescription):
	sample = tables.Int32Col(pos=0)
	epochs = tables.Int32Col(pos=1)
	loss = tables.Float32Col(pos=2)



#===============================================================
#------------------------ DATASET ------------------------------
#===============================================================
print 'Scanning Dataset:',dataset
with tables.open_file(dataset,'r') as cdbk:
	lines = cdbk.root.DATA.shape[0]

chosen = np.arange(lines)
random.shuffle(chosen)

DS={}
DS['train'] = chosen[0:int(0.8*lines)]		#80%
DS['valid'] = chosen[int(0.8*lines):]	#20%
#DS['test'] = chosen[int(0.8*lines):]
print 'Train ds length:', len(DS['train'])
print 'Valid ds length:', len(DS['valid'])



#===============================================================
#----------------------- Predictors ----------------------------
#===============================================================
for Pn, P in Predictors.iteritems():
	P.initialize(Name[Pn]+'.pt')
	loss_fn = torch.nn.MSELoss(size_average=False)			#Loss function: Mean Squared Error
	optimizer = torch.optim.SGD(P.parameters(), lr=1e-4)	#Optimzer: Stochastic Gradient Descend

	#Initialize counters
	best_run = 999999.0
	stage_loss = 0.0
	runin_loss = 0.0

#===============================================================
#---------------------- TRAIN LOOP -----------------------------
#===============================================================
	for epc in xrange(epochs):
		for stage in ['train', 'valid']:
			#reshuffle stage dataset
			random.shuffle(DS[stage])

			#Initialize counters
			since = time.time()
			stage_loss = 0.0
			runin_loss = 0.0

			for i, sample in enumerate(DS[stage]):
				#get sample from training_dataset
				with tables.open_file(dataset,'r') as cdbk:
					input_smpl = cdbk.root.DATA[sample]['features']
					out_T_smpl = cdbk.root.DATA[sample]['Translation']
					out_R_smpl = cdbk.root.DATA[sample]['euler_angles']
					output_smpl = np.concatenate((out_T_smpl.squeeze(),out_R_smpl.squeeze()))
				
				input_smpl = P.wrap_input(input_smpl)
				output_smpl = P.wrap_input(output_smpl)


				#Run predictor
				output = P(input_smpl)
				#Loss
				loss = loss_fn(output, output_smpl)
				stage_loss += loss.data[0]
				runin_loss += loss.data[0]


				#Backpropagation
				if stage == 'train':
					optimizer.zero_grad()	#Zero any previous gradients
					loss.backward()			#compute new grad
					optimizer.step()		#update weights in respect to the new grad, and SGD method in this case



				#save statiscs
				with tables.open_file(Name[Pn]+'.h5','a') as h5file:
					try: 	regist = h5file.create_table(h5file.root, stage, stats)
					except: 
						if stage == 'train' : 		regist = h5file.root.train
						elif stage == 'valid':	regist = h5file.root.valid

					regist.row['sample'] = sample
					regist.row['epochs'] = epc
					regist.row['loss'] = loss.data[0]
					regist.row.append()
					regist.flush					



				#CONTROL STATION
				if i % print_stats == print_stats-1:
					now = time.time() 
					duration = now - since
					duration_line = duration/print_stats
					since = now

					runin_loss = runin_loss/print_stats
					print "[pred_{} ,{:5} ,{:4} ,{:3}] loss:{:8.3f} | duration: {:.4f} s /line: {:.4f}  s".format(Pn, stage, epc, (i+1)/print_stats , runin_loss, duration, duration_line)
					runin_loss = 0


				#run only first 10 stats print
				if (i%(10*print_stats) == (10*print_stats)-1) and speed_run:
					break

			#===============================================================
			#---------------------- Validation -----------------------------
			#===============================================================
			stage_loss = stage_loss / (i+1)
			if stage == 'valid':			
				if stage_loss <= best_run:
					P.save(Name[Pn]+'.pt')
					best_run = stage_loss
					print 'Pred',Pn,'saved!  Best_run:', best_run
				print '-'*50
	#break	#only run predictor 0

