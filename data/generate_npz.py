from coffea.nanoaod import NanoEvents
import numpy as np
import os
from optparse import OptionParser
import concurrent.futures

def future_savez(i):
        #event = events[i]
        genmet_list = [
                events.GenMET.pt[i] * np.cos(events.GenMET.phi[i]),
                events.GenMET.pt[i] * np.sin(events.GenMET.phi[i]),
                events.MET.pt[i] * np.cos(events.MET.phi[i]),
                events.MET.pt[i] * np.sin(events.MET.phi[i]),
                events.PuppiMET.pt[i] * np.cos(events.PuppiMET.phi[i]),
                events.PuppiMET.pt[i] * np.sin(events.PuppiMET.phi[i]),
                events.DeepMETResponseTune.pt[i] * np.cos(events.DeepMETResponseTune.phi[i]),
                events.DeepMETResponseTune.pt[i] * np.sin(events.DeepMETResponseTune.phi[i]),
                events.DeepMETResolutionTune.pt[i] * np.cos(events.DeepMETResolutionTune.phi[i]),
                events.DeepMETResolutionTune.pt[i] * np.sin(events.DeepMETResolutionTune.phi[i])
        ]

        event_list = []
        n_particles=len(events.PFCands.pt[i])
        #print(n_particles)

        for j in range(n_particles):
                particle_list=[
                        events.PFCands.pt[i][j],
                        events.PFCands.eta[i][j],
                        events.PFCands.phi[i][j],
                        events.PFCands.mass[i][j],
                        events.PFCands.d0[i][j],
                        events.PFCands.dz[i][j],
                        events.PFCands.pdgId[i][j],
                        events.PFCands.charge[i][j],
                        events.PFCands.pvAssocQuality[i][j],
                        events.PFCands.puppiWeight[i][j]
                ]
                event_list.append(particle_list)
        npz_file=os.environ['PWD']+'/data/raw/'+dataset+'_event'+str(i)
        print('Saving file',npz_file+'.npz')
        return np.savez(npz_file,np.array(event_list),np.array(genmet_list))

if __name__ == '__main__':
        
        parser = OptionParser()
        parser.add_option('-d', '--dataset', help='dataset', dest='dataset')
        (options, args) = parser.parse_args()

        dataset=options.dataset
        fname = '/cms/scratch/matteoc/CMSSW_10_2_22/src/PhysicsTools/NanoMET/test/'+options.dataset+'.root'
        #fname = 'root://cms-xrdr.private.lo:2094//xrd/store/user/'+os.environ['USER']+'/'+dataset+'.root'
        print('Opening file:',fname)

        events = NanoEvents.from_file(fname)
        n_events=events.JetPFCands.pt.shape[0]
        print('Total events:',n_events)
        
        for i in range(n_events):
                future_savez(i)
        '''
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
                futures = set()
                futures.update(executor.submit(future_savez, i) for i in range(n_events))
                try:
                        total = len(futures)
                        processed = 0
                        while len(futures) > 0:
                                finished = set(job for job in futures if job.done())
                                for job in finished:
                                        job.result()
                                futures -= finished
                        del finished
                except KeyboardInterrupt:
                        print("Ok quitter")
                        for job in futures: job.cancel()
                except:
                        for job in futures: job.cancel()
                        raise
        '''
