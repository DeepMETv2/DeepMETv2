from coffea.nanoaod import NanoEvents
import numpy as np
from optparse import OptionParser
parser = OptionParser()
parser.add_option('-d', '--dataset', help='dataset', dest='dataset')
(options, args) = parser.parse_args()
fname = '/cms/scratch/matteoc/CMSSW_10_6_14/src/PhysicsTools/NanoMET/test/'+options.dataset+'.root'
events = NanoEvents.from_file(fname)
for i in range(events.JetPFCands.pt.shape[0]):
	event_list = []
	genmet_list = [events.GenMET.pt[i]]
	for j in range(len(events.JetPFCands.pt[i])):
		particle_list=[events.JetPFCands.pt[i][j],events.JetPFCands.eta[i][j],events.JetPFCands.phi[i][j],events.JetPFCands.mass[i][j]]
		event_list.append(particle_list)
	np.savez('/cms/scratch/matteoc/npz_files/'+options.dataset+'_event'+str(i),np.array(event_list),np.array(genmet_list))

