from coffea.nanoaod import NanoEvents
import numpy as np
import os
from optparse import OptionParser
import concurrent.futures

def future_savez(i, tot):
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
        n_particles=len(events.JetPFCands.pt[i])
        #print(n_particles)

        for j in range(n_particles):
                particle_list=[
                        events.JetPFCands.pt[i][j],
                        events.JetPFCands.eta[i][j],
                        events.JetPFCands.phi[i][j],
                        events.JetPFCands.mass[i][j],
                        events.JetPFCands.d0[i][j],
                        events.JetPFCands.dz[i][j],
                        events.JetPFCands.pdgId[i][j],
                        events.JetPFCands.charge[i][j],
                        events.JetPFCands.fromPV[i][j],
                        events.JetPFCands.puppiWeight[i][j],
                        events.JetPFCands.pvRef[i][j],
                        events.JetPFCands.pvAssocQuality[i][j]
                ]
                event_list.append(particle_list)
        npz_file=os.environ['PWD']+'/raw/'+dataset+'_event'+str(tot)
        print('Saving file',npz_file+'.npz')
        return np.savez(npz_file,np.array(event_list),np.array(genmet_list))

if __name__ == '__main__':
        
        parser = OptionParser()
        parser.add_option('-d', '--dataset', help='dataset', dest='dataset')
        parser.add_option('-s', '--startevt',type=int, default=1, help='startevt')
        (options, args) = parser.parse_args()
        datasetsname = {
            "znunu100to200": ['GraphMET_drop_trackinfocut/ZJetsToNuNu_HT-100To200_13TeV-madgraph/NanoAOD_0125/210226_233008',3],
            "znunu200to400": ['GraphMET_drop_trackinfocut/ZJetsToNuNu_HT-200To400_13TeV-madgraph/NanoAOD_0125/210226_233039',3],
            "znunu400to600": ['GraphMET_drop_trackinfocut/ZJetsToNuNu_HT-400To600_13TeV-madgraph/NanoAOD_0125/210226_233053',5],
            "znunu600to800": ['GraphMET_drop_trackinfocut/ZJetsToNuNu_HT-600To800_13TeV-madgraph/NanoAOD_0125/210226_233107',8],
            "znunu800to1200":[ 'GraphMET_drop_trackinfocut/ZJetsToNuNu_HT-800To1200_13TeV-madgraph/NanoAOD_0125/210226_233123',10],
            "znunu1200to2500": ['GraphMET_drop_trackinfocut/ZJetsToNuNu_HT-1200To2500_13TeV-madgraph/NanoAOD_0125/210226_233139',15],
            "znunu2500toInf": ['GraphMET_drop_trackinfocut/ZJetsToNuNu_HT-2500ToInf_13TeV-madgraph/NanoAOD_0125/210226_233153',30],
        }
        dataset=options.dataset
        tot=0
        start_n=1
        if(options.startevt > 1):
            start_n=options.startevt
        for i in range(1,datasetsname[dataset][1]+1):
            fname = 'root://cmseos.fnal.gov//store/user/yilai/'+datasetsname[dataset][0]+'/0000/output_nano_'+str(i) +'.root'
            print('Opening file:',fname)
            events = NanoEvents.from_file(fname)
            n_events=events.JetPFCands.pt.shape[0]
            print('N events:',n_events)
            print('Total events:',tot+n_events)
            for j in range(n_events):
                tot+=1
                if(tot>60000):
                    print("enough events>60000")
                    break
                else:
                    if(tot>start_n):
                        future_savez(j, tot)

        '''
        #fname = '/cms/scratch/matteoc/CMSSW_10_2_22/src/PhysicsTools/NanoMET/test/'+options.dataset+'.root'
        fname = 'root://cms-xrdr.private.lo:2094//xrd/store/user/'+os.environ['USER']+'/'+dataset+'.root'
        #fname = 'root://cms-xrdr.private.lo:2094//xrd/store/user/'+os.environ['USER']+'/'+dataset+'.root'
        print('Opening file:',fname)

        events = NanoEvents.from_file(fname)
        n_events=events.JetPFCands.pt.shape[0]
        print('Total events:',n_events)
        
        for i in range(n_events):
                future_savez(i)
        
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
