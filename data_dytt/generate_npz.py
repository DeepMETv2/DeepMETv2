"""
script to generate npz files with event and PF candidates selections
similar to DeepMET studies
"""

from coffea.nanoaod import NanoEvents
import numpy as np
import os
from optparse import OptionParser
import concurrent.futures

def DeltaR2(eta1, phi1, eta2, phi2):
    deta2 = (eta1 - eta2)**2
    dphi = phi1 - phi2
    if dphi < -np.pi:
        dphi += 2 * np.pi
    elif dphi > np.pi:
        dphi -= 2 * np.pi
    return deta2 + dphi**2


def future_savez(i, tot, nfile):
    muons = events_selected.Muon[events_selected.Muon.istight]
    electrons = events_selected.Electron[events_selected.Electron.istight]
    leptons = []
    leptons_px = 0.
    leptons_py = 0.
    for ilep in range(options.n_leptons_subtract):
        if ilep < muons[i].size:
            leptons_px += muons.pt[i][ilep] * np.cos(muons.phi[i][ilep])
            leptons_py += muons.pt[i][ilep] * np.sin(muons.phi[i][ilep])
            leptons.append(muons[i][ilep])
        if ilep < electrons[i].size:
            leptons_px += electrons.pt[i][ilep] * np.cos(electrons.phi[i][ilep])
            leptons_py += electrons.pt[i][ilep] * np.sin(electrons.phi[i][ilep])
            leptons.append(electrons[i][ilep])

    genmet_list = [
        events_selected.GenMET.pt[i] * np.cos(events_selected.GenMET.phi[i]) + leptons_px,
        events_selected.GenMET.pt[i] * np.sin(events_selected.GenMET.phi[i]) + leptons_py,
        events_selected.MET.pt[i] * np.cos(events_selected.MET.phi[i]) + leptons_px,
        events_selected.MET.pt[i] * np.sin(events_selected.MET.phi[i]) + leptons_py,
        events_selected.PuppiMET.pt[i] * np.cos(events_selected.PuppiMET.phi[i]) + leptons_px,
        events_selected.PuppiMET.pt[i] * np.sin(events_selected.PuppiMET.phi[i]) + leptons_py,
        events_selected.DeepMETResponseTune.pt[i] * np.cos(events_selected.DeepMETResponseTune.phi[i]) + leptons_px,
        events_selected.DeepMETResponseTune.pt[i] * np.sin(events_selected.DeepMETResponseTune.phi[i]) + leptons_py,
        events_selected.DeepMETResolutionTune.pt[i] * np.cos(events_selected.DeepMETResolutionTune.phi[i]) + leptons_px,
        events_selected.DeepMETResolutionTune.pt[i] * np.sin(events_selected.DeepMETResolutionTune.phi[i]) + leptons_py
    ]

    event_list = []
    n_particles = len(events_selected.JetPFCands.pt[i])
    #print('Event:',i,'number of PF candidates:',n_particles)
    for j in range(n_particles):
        islepton = False
        for jlep in range(options.n_leptons_subtract):
            dr2 = DeltaR2(events_selected.JetPFCands.eta[i][j], events_selected.JetPFCands.phi[i][j], leptons[jlep].eta, leptons[jlep].phi)
            if dr2 < 0.0001:
                islepton = True
                break
        if not islepton:
            particle_list = [
                events_selected.JetPFCands.pt[i][j],
                events_selected.JetPFCands.eta[i][j],
                events_selected.JetPFCands.phi[i][j],
                events_selected.JetPFCands.mass[i][j],
                events_selected.JetPFCands.d0[i][j],
                events_selected.JetPFCands.dz[i][j],
                events_selected.JetPFCands.pdgId[i][j],
                events_selected.JetPFCands.charge[i][j],
                events_selected.JetPFCands.fromPV[i][j],
                events_selected.JetPFCands.puppiWeight[i][j],
                events_selected.JetPFCands.pvRef[i][j],
                events_selected.JetPFCands.pvAssocQuality[i][j],
            ]
            event_list.append(particle_list)
        #else:
        #    print ("jlep: ", jlep, " dr2 ", dr2)
    if(nfile==-1):
        npz_file=os.environ['PWD']+'/raw/'+dataset+'_event'+str(tot)
    else:
        npz_file=os.environ['PWD']+'/raw/'+dataset+str(nfile)+'_event'+str(tot)
    print('Saving file',npz_file+'.npz')
    return np.savez_compressed(npz_file,np.array(event_list),np.array(genmet_list))

def SelectEvent(nlepcut):
    # select muons
    select_tight_muon = ((events.Muon.tightId == 1) & (
        events.Muon.pfRelIso03_all < 0.15) & (events.Muon.pt > 20.))
    #muons = events.Muon[select_tight_muon]
    events.Muon['istight']=select_tight_muon
    muons=events.Muon[events.Muon.istight]

    # select electrons
    select_tight_electron = (
        (events.Electron.mvaFall17V1Iso_WP80 == 1) & (events.Electron.pt > 20.0))
    #electrons = events.Electron[select_tight_electron]
    events.Electron['istight']=select_tight_electron
    electrons=events.Electron[events.Electron.istight]

    nlep = muons.counts + electrons.counts

    events_selected = events[nlep == nlepcut]

    return events_selected


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option('-d', '--dataset', help='dataset', dest='dataset', default='Test')
    parser.add_option('-s', '--startevt',type=int, default=0, help='start event')
    parser.add_option('-f', '--file_number',type=int, default=-1, help='file number')
    parser.add_option('-n', '--maxNumberr',type=int, default=-1, help='events number')
    parser.add_option('--n_leptons', dest='n_leptons',
                      help='How many leptons are required in the events', default=2)
    parser.add_option('--n_leptons_subtract', dest='n_leptons_subtract',
                      help='How many leptons to be subtracted from the Candidates list. Can not be larger than the n_leptons', default=2)

    (options, args) = parser.parse_args()
    assert options.n_leptons >= options.n_leptons_subtract, "n_leptons_subtract can not be larger than n_leptons"
    datasetsname = {
        "dy": ['GraphMET_drop_trackinfocut/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/NanoAOD_0125/210227_040343/',10],
        "tt": ['GraphMET_drop_trackinfocut/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8/NanoAOD_0125/210227_040403/',10],
    }
    dataset=options.dataset
    tot=0
    tot_target=1
    if(options.maxNumberr>tot_target):
        tot_target=options.maxNumberr
    start_n=0
    if(options.startevt > 0):
        start_n=options.startevt
    if(options.file_number==-1):
        for i in range(1,datasetsname[dataset][1]+1):
            fname = 'root://cmseos.fnal.gov//store/user/yilai/'+datasetsname[dataset][0]+'/0000/output_nano_'+str(i) +'.root'
            print('Opening file:',fname)
            events = NanoEvents.from_file(fname)
            events_selected = SelectEvent( options.n_leptons )
            n_events = events_selected.JetPFCands.pt.shape[0]
            print('Total events:', n_events)
            for j in range(n_events):
                tot+=1
                if(tot>tot_target):
                    print("enough events ", tot)
                    break
                else:
                    if(tot>start_n):
                        future_savez(j, tot, options.file_number)
            if(tot>tot_target):
                print("enough events ", tot-1)
                break
        print("finished")
    else:
        fname = 'root://cmseos.fnal.gov//store/user/yilai/'+datasetsname[dataset][0]+'/0000/output_nano_'+str(options.file_number) +'.root'
        print('Opening file:',fname)
        events = NanoEvents.from_file(fname)
        events_selected = SelectEvent( options.n_leptons )
        n_events=events_selected.JetPFCands.pt.shape[0]
        print('N events:',n_events)
        print('Total events:',tot+n_events)
        for j in range(n_events):
            tot+=1
            if(tot>tot_target):
                print("enough events ", tot-1)
                break
            else:
                if(tot>start_n):
                    future_savez(j, tot, options.file_number)
        print("finished")


    
    '''
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
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
