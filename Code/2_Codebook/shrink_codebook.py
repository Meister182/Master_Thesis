#! /usr/bin/env python
#================================   IMPORTS   ==================================
import tables
import time
import numpy as np





#===============================================================
#------------------------ Configs ------------------------------
#===============================================================
#codebook file to be shrinked:
codebook = 'codebook.h5'

new_codebook ='new_codebook.h5' 
percent = 10 	#extract 10 percent of original


get_line = int(100 / percent)
print_percent = 0
this_line = 0

#===============================================================
#------------------------- Format ------------------------------
#===============================================================
class synth_patch(tables.IsDescription):
    features = tables.Float32Col(pos=1, shape=[1,256])       
    Translation = tables.Float32Col(pos=2, shape=[1,3])
    euler_angles = tables.Float32Col(pos=3, shape=[1,3])
    obj_ids = tables.Int32Col(pos=4)




#===============================================================
#---------------------- Lines ----------------------------------
#===============================================================
print 'Scanning OG codebook:', codebook
with tables.open_file(codebook,'r') as cdbk:
	lines = cdbk.root.DATA.shape[0]

new_lines = lines * (percent/100.)

print 'OG  codebook length:', lines, 'lines.'
print 'New codebook length:', new_lines, 'lines.'


since = time.time()
for line in xrange(lines):
	if line % get_line == 0:
		#Get line
		with tables.open_file(codebook, mode='r') as cdbk:
			feat = cdbk.root.DATA[line]['features']
			tran = cdbk.root.DATA[line]['Translation']
			orie = cdbk.root.DATA[line]['euler_angles']
			obID = cdbk.root.DATA[line]['obj_ids']


		with tables.open_file(new_codebook, mode='a') as h5file:
			try: 	cbk = h5file.create_table(h5file.root, 'DATA', synth_patch)
			except: cbk = h5file.root.DATA
			cbk.row['features'] 	= feat
			cbk.row['Translation'] 	= tran
			cbk.row['euler_angles'] = orie
			cbk.row['obj_ids'] 		= obID
			cbk.row.append()
			cbk.flush



	# Print stats
	if int(100 * line/lines) != print_percent:

		now = time.time()
		print_percent = int((100 * line/lines))
		print print_percent,'percent completed.   time, till now:', now-since, '  /line:{:5f}'.format((now-since)/(line-this_line))
		this_line = line
		since = now








