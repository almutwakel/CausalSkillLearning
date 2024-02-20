cd /scratch/cchawla/TrainingLogs/HumanDatasetAnalysis_SwappedObjects

indices = np.concatenate([np.arange(7,10), np.arange(0,3)])
# indices = np.concatenate([np.arange(14,17), np.arange(0,3)])
# indices = np.arange(7,10)

for k in range(len(self.dataset)):
	print("##", k)    
	ax = plt.gca()
	ax.set_ylim([-1,1])
	plt.plot(self.dataset[k]['demo'][:,indices], 'o')
	# plt.plot(self.dataset[k]['demo'][:,7:10])
	
	plt.savefig("Traj_{0}.png".format(str(k).zfill(2)))
	plt.close()

# Open Results HTML file. 	    
with open('Results.html','w') as html_file:
	
	# Start HTML doc. 
	html_file.write('<html>')
	html_file.write('<body>')
	# html_file.write('<p> Model: {0}</p>'.format(self.args.name))						
	# html_file.write('<p> Average Trajectory Distance: {0}</p>'.format(self.mean_distance))


	for i in range(len(self.dataset)):
				
		print("Datapoint:",i)                        
		html_file.write('<p> <b> Trajectory {}  </b></p>'.format(i))

		# file_prefix = self.dir_name

		html_file.write('<div style="display: flex; justify-content: row;">  <img src="Traj_{0}.png"/> </div>'.format(str(i).zfill(2)))
			
		# Add gap space.
		html_file.write('<p> </p>')

	html_file.write('</body>')
	html_file.write('</html>')






for k in range(len(self.dataset)):
# for k in range(11):
	first_obj_state = self.dataset[k]['dummy_object_state'][...,:3]
	second_obj_state = self.dataset[k]['dummy_object_state'][...,7:10]

	max_o1 = np.max(abs(np.diff(first_obj_state, axis=0)))
	max_o2 = np.max(abs(np.diff(second_obj_state, axis=0)))
	tid = self.dataset[k]['task-id']
	eid = self.dataset.environment_names[tid]
	print("######################################")	
	print(k, tid, eid, max_o1, max_o2)
