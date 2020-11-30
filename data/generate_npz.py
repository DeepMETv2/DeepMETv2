from coffea.nanoaod import NanoEvents
import numpy as np
import os
from optparse import OptionParser
import concurrent.futures

def future_savez(events, i):
        event = events[i]

        genmet_list = [
                event.GenMET.pt * np.cos(event.GenMET.phi),
                event.GenMET.pt * np.sin(event.GenMET.phi),
                event.MET.pt * np.cos(event.MET.phi),
                event.MET.pt * np.sin(event.MET.phi),
                event.PuppiMET.pt * np.cos(event.PuppiMET.phi),
                event.PuppiMET.pt * np.sin(event.PuppiMET.phi),
                event.DeepMETResponseTune.pt * np.cos(event.DeepMETResponseTune.phi),
                event.DeepMETResponseTune.pt * np.sin(event.DeepMETResponseTune.phi),
                event.DeepMETResolutionTune.pt * np.cos(event.DeepMETResolutionTune.phi),
                event.DeepMETResolutionTune.pt * np.sin(event.DeepMETResolutionTune.phi)
        ]

        event_list = []
        n_particles=len(events.JetPFCands.pt[i])

        for j in range(n_particles):
                particle_list=[
                        event.JetPFCands.pt[j],
                        event.JetPFCands.eta[j],
                        event.JetPFCands.phi[j],
                        event.JetPFCands.mass[j],
                        event.JetPFCands.d0[j],
                        event.JetPFCands.dz[j],
                        event.JetPFCands.pdgId[j],
                        event.JetPFCands.charge[j],
                        event.JetPFCands.pvAssocQuality[j],
                        event.JetPFCands.puppiWeight[j]
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
        #fname = '/cms/scratch/matteoc/CMSSW_10_2_22/src/PhysicsTools/NanoMET/test/'+options.dataset+'.root'
        fname = 'root://cms-xrdr.private.lo:2094//xrd/store/user/'+os.environ['USER']+'/'+dataset+'.root'
        print('Opening file:',fname)

        events = NanoEvents.from_file(fname)
        n_events=events.JetPFCands.pt.shape[0]
        print('Total events:',n_events)
        
        for i in range(n_events):
                future_savez(events, i)
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
