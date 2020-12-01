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


def future_savez(i):
    #event = events_selected[i]
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
        #else:
        if ilep < electrons[i].size:
            leptons_px += electrons.pt[i][ilep] * np.cos(electrons.phi[i][ilep])
            leptons_py += electrons.pt[i][ilep] * np.sin(electrons.phi[i][ilep])
            leptons.append(electrons[i][ilep])
    #print(leptons_px, leptons_py, muons_selected[i].size, electrons_selected[i].size)

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
                events_selected.JetPFCands.pvAssocQuality[i][j],
                events_selected.JetPFCands.puppiWeight[i][j]
            ]
            event_list.append(particle_list)
        #else:
        #    print ("jlep: ", jlep, " dr2 ", dr2)

    npz_file = os.environ['PWD']+'/data_dytt/raw/'+dataset+'_event'+str(i)
    print('Saving file', npz_file+'.npz')
    return np.savez(npz_file, np.array(event_list), np.array(genmet_list))


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
    parser.add_option('--n_leptons', dest='n_leptons',
                      help='How many leptons are required in the events', default=2)
    parser.add_option('--n_leptons_subtract', dest='n_leptons_subtract',
                      help='How many leptons to be subtracted from the Candidates list. Can not be larger than the n_leptons', default=2)
    (options, args) = parser.parse_args()
    dataset=options.dataset

    assert options.n_leptons >= options.n_leptons_subtract, "n_leptons_subtract can not be larger than n_leptons"

    fname = 'root://cms-xrdr.private.lo:2094//xrd/store/user/'+os.environ['USER']+'/'+dataset+'.root'
    print('Opening file:', fname)

    events = NanoEvents.from_file(fname)

    events_selected = SelectEvent( options.n_leptons )
    n_events = events_selected.JetPFCands.pt.shape[0]
    print('Total events:', n_events)

    
    for i in range(n_events):
    #for i in range(10):
        future_savez(i)
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
