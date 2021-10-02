"""
script to generate npz files with event and PF candidates selections
similar to DeepMET studies
"""
from coffea.nanoevents import NanoEventsFactory
from coffea.nanoevents.schemas import NanoAODSchema,BaseSchema
import numpy as np
import os
from optparse import OptionParser
import concurrent.futures
import glob
import awkward as ak
import time
import json
from collections import OrderedDict,defaultdict
recdd = lambda : defaultdict(recdd) ## define recursive defaultdict
JSON_LOC = 'filelist.json'


def multidict_tojson(filepath, indict):
    ## expand into multidimensional dictionary
    with open(filepath, "w") as fo:
        json.dump( indict, fo)
        print("save to %s" %filepath)

def delta_phi(obj1, obj2):
    return (obj1.phi - obj2.phi + np.pi) % (2 * np.pi) - np.pi

def delta_r(obj1, obj2):
    return np.sqrt((obj1.eta - obj2.eta) ** 2 + delta_phi(obj1, obj2) ** 2)

def run_deltar_matching(store,
                        target,
                        drname='deltaR',
                        radius=0.4,
                        unique=False,
                        sort=False):
  """
  Running a delta R matching of some object collection "store" of dimension NxS
  with some target collection "target" of dimension NxT, The return object will
  have dimension NxSxT' where objects in the T' contain all "target" objects
  within the delta R radius. The delta R between the store and target object will
  be stored in the field `deltaR`. If the unique flag is turned on, then objects
  in the target collection will only be associated to the closest object. If the
  sort flag is turned on, then the target collection will be sorted according to
  the computed `deltaR`.
  """
  _, target = ak.unzip(ak.cartesian([store.eta, target], nested=True))
  target[drname] = delta_r(store, target)
  if unique:  # Additional filtering
    t_index = ak.argmin(target[drname], axis=-2)
    s_index = ak.local_index(store.eta, axis=-1)
    _, t_index = ak.unzip(ak.cartesian([s_index, t_index], nested=True))
    target = target[s_index == t_index]

  # Cutting on the computed delta R
  target = target[target[drname] < radius]

  # Sorting according to the computed delta R
  if sort:
    idx = ak.argsort(target[drname], axis=-1)
    target = target[idx]
  return target


def future_savez(dataset,currentfile):

    print('before selection ', len(events_slice))
    # select Muon
    myMuon = events_slice.Muon[:]
    myMuon['istight'] = ((events_slice.Muon.tightId == 1) & ( events_slice.Muon.pfRelIso03_all < 0.15) & (events_slice.Muon.pt > 20.))
    events_slice['Muon'] = myMuon[myMuon.istight]
    # select electrons
    myElectron = events_slice.Electron[:]
    myElectron['istight'] = ((events_slice.Electron.mvaFall17V1Iso_WP80 == 1) & (events_slice.Electron.pt > 20.0))
    events_slice['Electron'] = myElectron[myElectron.istight]
    # select events with n tight leptons
    n_tight_leptons = ak.count(events_slice.Muon.pt[events_slice.Muon.istight],axis=-1)+ak.count(events_slice.Electron.pt[events_slice.Electron.istight],axis=-1)
    # number of leptons can be larger than the required number 
    events_selected = events_slice[n_tight_leptons >= options.n_leptons]
    print('after selection ', len(events_selected))
    
    muons = events_selected.Muon[events_selected.Muon.istight]
    electrons = events_selected.Electron[events_selected.Electron.istight]
    # mix leptons and sort according to pt
    leptons = ak.concatenate( [muons, electrons], axis=1) 
    leptons = leptons[ak.argsort(leptons.pt,axis=1,ascending=False)]
    leptons =  leptons[:,0:int(options.n_leptons_subtract)]
    # only want the first n_leptons_subtract leptons
    #print('number of leptons ', ak.count(leptons.pt, axis=-1))
    leptons_px = leptons.pt * np.cos(leptons.phi)
    leptons_py = leptons.pt * np.sin(leptons.phi)
    leptons_px = ak.sum(leptons_px,axis=1)
    leptons_py = ak.sum(leptons_py,axis=1)
    met_list = np.column_stack([
            events_selected.GenMET.pt * np.cos(events_selected.GenMET.phi)+ leptons_px,
            events_selected.GenMET.pt * np.sin(events_selected.GenMET.phi)+ leptons_py,
            events_selected.MET.pt * np.cos(events_selected.MET.phi)+ leptons_px,
            events_selected.MET.pt * np.sin(events_selected.MET.phi)+ leptons_py,
            events_selected.PuppiMET.pt * np.cos(events_selected.PuppiMET.phi)+ leptons_px,
            events_selected.PuppiMET.pt * np.sin(events_selected.PuppiMET.phi)+ leptons_py,
            events_selected.DeepMETResponseTune.pt * np.cos(events_selected.DeepMETResponseTune.phi)+ leptons_px,
            events_selected.DeepMETResponseTune.pt * np.sin(events_selected.DeepMETResponseTune.phi)+ leptons_py,
            events_selected.DeepMETResolutionTune.pt * np.cos(events_selected.DeepMETResolutionTune.phi)+ leptons_px,
            events_selected.DeepMETResolutionTune.pt * np.sin(events_selected.DeepMETResolutionTune.phi)+ leptons_py,
            events_selected.LHE.HT
    ])
    overlap_removal = run_deltar_matching(events_selected.PFCands,
                        leptons,
                        drname='deltaR',
                        radius=0.001,
                        unique=True,
                        sort=False)
    # remove the cloest PF particle 
    mask = ak.count(overlap_removal.deltaR,axis=-1)==0
    #print(len(events_selected.PFCands.pt[0]))
    events_selected['PFCands']=events_selected.PFCands[mask]
    #print(len(events_selected.PFCands.pt[0]))
    #save the rest of PFcandidates 


    nparticles_per_event = max(ak.num(events_selected.PFCands.pt, axis=1))
    print("max NPF in this range: ", nparticles_per_event)

    particle_list = ak.concatenate([
                 [ ak.fill_none(ak.pad_none(events_selected.PFCands.pt, nparticles_per_event,clip=True),-999)           ] ,
                 [ ak.fill_none(ak.pad_none(events_selected.PFCands.eta, nparticles_per_event,clip=True),-999)          ] ,
                 [ ak.fill_none(ak.pad_none(events_selected.PFCands.phi, nparticles_per_event,clip=True),-999)          ] ,
                 [ ak.fill_none(ak.pad_none(events_selected.PFCands.d0, nparticles_per_event,clip=True),-999)           ] ,
                 [ ak.fill_none(ak.pad_none(events_selected.PFCands.dz, nparticles_per_event,clip=True),-999)           ] ,
                 [ ak.fill_none(ak.pad_none(events_selected.PFCands.mass, nparticles_per_event,clip=True),-999)         ] ,
                 [ ak.fill_none(ak.pad_none(events_selected.PFCands.puppiWeight, nparticles_per_event,clip=True),-999)  ] ,
                 [ ak.fill_none(ak.pad_none(events_selected.PFCands.pdgId, nparticles_per_event,clip=True),-999)        ] ,
                 [ ak.fill_none(ak.pad_none(events_selected.PFCands.charge, nparticles_per_event,clip=True),-999)        ] ,
                 [ ak.fill_none(ak.pad_none(events_selected.PFCands.fromPV, nparticles_per_event,clip=True),-999)        ] ,
                 [ ak.fill_none(ak.pad_none(events_selected.PFCands.pvRef, nparticles_per_event,clip=True),-999)         ] ,
                 [ ak.fill_none(ak.pad_none(events_selected.PFCands.pvAssocQuality, nparticles_per_event,clip=True),-999)] ,
    ])
    npz_file=os.environ['PWD']+'/raw/'+dataset+'_file'+str(currentfile)+'_slice_'+str(i)+'_nevent_'+str(len(events_selected))
    np.savez(npz_file,x=particle_list,y=met_list)



if __name__ == '__main__':

        parser = OptionParser()
        parser.add_option('-d', '--dataset', help='dataset', dest='dataset')
        parser.add_option('-s', '--startfile',type=int, default=0, help='startfile')
        parser.add_option('-e', '--endfile',type=int, default=1, help='endfile')
        parser.add_option('--n_leptons', dest='n_leptons',
                          help='How many leptons are required in the events', default=2)
        parser.add_option('--n_leptons_subtract', dest='n_leptons_subtract',
                          help='How many leptons to be subtracted from the Candidates list. Can not be larger than the n_leptons', default=2)
        (options, args) = parser.parse_args()

        assert options.n_leptons >= options.n_leptons_subtract, "n_leptons_subtract can not be larger than n_leptons"
        datasetsname = {
            "dy": ['DYJetsToLL/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8'],
            "tt": ['TTTo2L2Nu/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/'],
        }
        # Be nice to eos, save list to a file
        #filelists = recdd()
        #for datset in datasetsname.keys():
        #    filelists[datset] = glob.glob('/eos/uscms/store/group/lpcjme/NanoMET/'+datasetsname[datset][0]+'/*/*/*/*root')
        #    filelists[datset] = [x.replace('/eos/uscms','root://cmseos.fnal.gov/') for x in filelists[datset] ]
        #multidict_tojson(JSON_LOC, filelists )
        #exit()
        dataset=options.dataset
        if dataset not in datasetsname.keys():
            print('choose one of them: ', datasetsname.keys())
            exit()
        #Read file from json
        with open(JSON_LOC, "r") as fo:
            file_names = json.load(fo)
        file_names = file_names[dataset]
        print('find ', len(file_names)," files")
        if options.startfile>=options.endfile and options.endfile!=-1:
            print("make sure options.startfile<options.endfile")
            exit()
        inpz=0
        eventperfile=5000
        currentfile=0
        for ifile in file_names:
            if currentfile<options.startfile:
                currentfile+=1
                continue
            events = NanoEventsFactory.from_root(ifile, schemaclass=NanoAODSchema).events()
            nevents_total = len(events)
            print(ifile, ' Number of events:', nevents_total)

            for i in range(int(nevents_total / eventperfile)+1):
                if i< int(nevents_total / eventperfile):
                    print('from ',i*eventperfile, ' to ', (i+1)*eventperfile)
                    events_slice = events[i*eventperfile:(i+1)*eventperfile]
                elif i == int(nevents_total / eventperfile) and i*eventperfile<=nevents_total:
                    print('from ',i*eventperfile, ' to ', nevents_total)
                    events_slice = events[i*eventperfile:nevents_total]
                else:
                    print(' weird ... ')
                    exit()
                tic=time.time()
                future_savez(dataset,currentfile)
                toc=time.time()
                print('time:',toc-tic)
            currentfile+=1
            if currentfile>=options.endfile:
                print('=================> finished ')
                exit()

