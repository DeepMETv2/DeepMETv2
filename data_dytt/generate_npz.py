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


def future_savez(events_selected, i):
    event = events_selected[i]
    muons = event.Muon[event.Muon.istight]
    electrons = event.Electron[event.Electron.istight]
    leptons = []
    leptons_px = 0.
    leptons_py = 0.
    for ilep in range(options.n_leptons_subtract):
        if ilep < muons.size:
            leptons_px += muons.pt[ilep] * np.cos(muons.phi[ilep])
            leptons_py += muons.pt[ilep] * np.sin(muons.phi[ilep])
            leptons.append(muons[ilep])
        #else:
        if ilep < electrons.size:
            leptons_px += electrons.pt[ilep] * np.cos(electrons.phi[ilep])
            leptons_py += electrons.pt[ilep] * np.sin(electrons.phi[ilep])
            leptons.append(electrons[ilep])
    #print(leptons_px, leptons_py, muons_selected[i].size, electrons_selected[i].size)

    genmet_list = [
        event.GenMET.pt * np.cos(event.GenMET.phi) + leptons_px,
        event.GenMET.pt * np.sin(event.GenMET.phi) + leptons_py,
        event.MET.pt * np.cos(event.MET.phi) + leptons_px,
        event.MET.pt * np.sin(event.MET.phi) + leptons_py,
        event.PuppiMET.pt * np.cos(event.PuppiMET.phi) + leptons_px,
        event.PuppiMET.pt * np.sin(event.PuppiMET.phi) + leptons_py,
        event.DeepMETResponseTune.pt * np.cos(event.DeepMETResponseTune.phi) + leptons_px,
        event.DeepMETResponseTune.pt * np.sin(event.DeepMETResponseTune.phi) + leptons_py,
        event.DeepMETResolutionTune.pt * np.cos(event.DeepMETResolutionTune.phi) + leptons_px,
        event.DeepMETResolutionTune.pt * np.sin(event.DeepMETResolutionTune.phi) + leptons_py
    ]

    event_list = []
    n_particles = len(event.JetPFCands.pt)
    #print('Event:',i,'number of PF candidates:',n_particles)
    for j in range(n_particles):
        islepton = False
        for jlep in range(options.n_leptons_subtract):
            dr2 = DeltaR2(event.JetPFCands.eta[j], event.JetPFCands.phi[j], leptons[jlep].eta, leptons[jlep].phi)
            if dr2 < 0.0001:
                islepton = True
                break
        if not islepton:
            particle_list = [
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
        future_savez(events_selected, i)
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
