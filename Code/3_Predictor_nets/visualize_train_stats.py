#! /usr/bin/env python
#================================   IMPORTS   ==================================
import matplotlib.pyplot as plt
import tables
import numpy as np



#===============================================================
#------------------------ Configs ------------------------------
#===============================================================
Names = ['Predictor_0','Predictor_1','Predictor_2','Predictor_3']
Names = ['Predictor_0']

fig, axes = plt.subplots(1,2)
#mng = plt.get_current_fig_manager()
#mng.window.showMaximized()



for Name in Names:
	fig.suptitle(Name+' Results', fontsize=14, fontweight='bold')

	axes[0].set_title(Name+' Train Loss')
	axes[0].grid(True)

	axes[1].set_title(Name+' Valid Loss')
	axes[1].grid(True)


#===============================================================
#--------------------- Load stats ------------------------------
#===============================================================
	print Name, "loading stats.."
	with tables.open_file(Name+'.h5', 'r') as regist:
		train_epcs = regist.root.train[:]['epochs']
		train_loss = regist.root.train[:]['loss']

		valid_epcs = regist.root.valid[:]['epochs']
		valid_loss = regist.root.valid[:]['loss']


	#Find epochs indices
	def find_epoch(epochs):
		epcs=[]
		for i, ep in enumerate(epochs[:-1]):
			if i == 0:	epcs.append(i)
			if not ep == epochs[i+1]:
				epcs.append(i+1)
		return epcs

	train_epc_i = find_epoch(train_epcs)
	valid_epc_i = find_epoch(valid_epcs)



#===============================================================
#------------------------ Plotter ------------------------------
#===============================================================
	#Train Stats
	axes[0].plot(np.arange(len(train_loss)), train_loss)
	for ep in train_epc_i:
		axes[0].axvline(x=ep, color='r')
		axes[0].text(ep+0.1,0,'epoch'+str(train_epcs[ep]),rotation=90)

	#Valid Stats
	axes[1].plot(np.arange(len(valid_loss)), valid_loss)
	for ep in valid_epc_i:
		axes[1].axvline(x=ep, color='r')
		axes[1].text(ep+0.1,0,'epoch'+str(valid_epcs[ep]),rotation=90)

	plt.pause(0.1)

	print 'Enter: to Save\nQ/q: quit without saving'
	opt = raw_input('::')
	if opt in ['q','Q']:
		break

	plt.savefig(Name+"_results.png")
	axes[0].clear()
	axes[1].clear()


