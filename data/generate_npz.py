from coffea.nanoaod import NanoEvents
import numpy as np
import os
from optparse import OptionParser
import concurrent.futures

def future_savez(i):
        event_list = []
        genmet_list = [events.GenMET.pt[i],events.GenMET.phi[i]]
        n_particles=len(events.JetPFCands.pt[i])
        print('Event:',i,'number of PF candidates:',n_particles)
        for j in range(n_particles):
                particle_list=[events.JetPFCands.pt[i][j],events.JetPFCands.eta[i][j],events.JetPFCands.phi[i][j],events.JetPFCands.mass[i][j]]
                event_list.append(particle_list)
        npz_file=os.environ['PWD']+'/data/raw/'+dataset+'_event'+str(i)
        print('Saving file',npz_file+'.npz')
        return np.savez(npz_file,np.array(event_list),np.array(genmet_list))

if __name__ == '__main__':
        
        parser = OptionParser()
        parser.add_option('-d', '--dataset', help='dataset', dest='dataset')
        (options, args) = parser.parse_args()

        dataset=options.dataset
        #fname = '/cms/scratch/matteoc/CMSSW_10_6_14/src/PhysicsTools/NanoMET/test/'+options.dataset+'.root'
        fname = 'root://cms-xrdr.private.lo:2094//xrd/store/user/'+os.environ['USER']+'/'+dataset+'.root'
        print('Opening file:',fname)

        events = NanoEvents.from_file(fname)
        n_events=events.JetPFCands.pt.shape[0]
        print('Total events:',n_events)

        with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
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
