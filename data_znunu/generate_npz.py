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

def future_savez(i, tot):
        #tic=time.time()
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
                events.DeepMETResolutionTune.pt[i] * np.sin(events.DeepMETResolutionTune.phi[i]),
                events.LHE.HT[i]
        ]

        particle_list = np.column_stack([
                      events.PFCands.pt[i],
                      events.PFCands.eta[i],
                      events.PFCands.phi[i],
                      events.PFCands.mass[i],
                      events.PFCands.d0[i],
                      events.PFCands.dz[i],
                      events.PFCands.pdgId[i],
                      events.PFCands.charge[i],
                      events.PFCands.fromPV[i],
                      events.PFCands.puppiWeight[i],
                      events.PFCands.pvRef[i],
                      events.PFCands.pvAssocQuality[i]
        ])
        eventi = [particle_list,genmet_list]
        #toc=time.time()
        #print(toc-tic)
        return eventi


if __name__ == '__main__':
        
        parser = OptionParser()
        parser.add_option('-d', '--dataset', help='dataset', dest='dataset')
        parser.add_option('-s', '--startfile',type=int, default=0, help='startfile')
        parser.add_option('-e', '--endfile',type=int, default=1, help='endfile')
        (options, args) = parser.parse_args()
        datasetsname = {
            "znunu200to400": ['Znunu/ZJetsToNuNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8/'],
            "znunu400to600": ['Znunu/ZJetsToNuNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8/'],
            "znunu600to800": ['Znunu/ZJetsToNuNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8/'],
            "znunu800to1200":[ 'Znunu/ZJetsToNuNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8/'],
            "znunu1200to2500": ['Znunu/ZJetsToNuNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8/'],
            "znunu2500toInf": ['Znunu/ZJetsToNuNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8/'],
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
        eventperfile=1000
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

                nparticles_per_event = max(ak.num(events_slice.PFCands.pt, axis=1))
                print("max NPF in this range: ", nparticles_per_event)
                tic=time.time()
                met_list = np.column_stack([
                        events_slice.GenMET.pt * np.cos(events_slice.GenMET.phi),
                        events_slice.GenMET.pt * np.sin(events_slice.GenMET.phi),
                        events_slice.MET.pt * np.cos(events_slice.MET.phi),
                        events_slice.MET.pt * np.sin(events_slice.MET.phi),
                        events_slice.PuppiMET.pt * np.cos(events_slice.PuppiMET.phi),
                        events_slice.PuppiMET.pt * np.sin(events_slice.PuppiMET.phi),
                        events_slice.DeepMETResponseTune.pt * np.cos(events_slice.DeepMETResponseTune.phi),
                        events_slice.DeepMETResponseTune.pt * np.sin(events_slice.DeepMETResponseTune.phi),
                        events_slice.DeepMETResolutionTune.pt * np.cos(events_slice.DeepMETResolutionTune.phi),
                        events_slice.DeepMETResolutionTune.pt * np.sin(events_slice.DeepMETResolutionTune.phi),
                        events_slice.LHE.HT
                ])
                particle_list = ak.concatenate([
                             [ ak.fill_none(ak.pad_none(events_slice.PFCands.pt, nparticles_per_event,clip=True),-999)           ] ,
                             [ ak.fill_none(ak.pad_none(events_slice.PFCands.eta, nparticles_per_event,clip=True),-999)          ] ,
                             [ ak.fill_none(ak.pad_none(events_slice.PFCands.phi, nparticles_per_event,clip=True),-999)          ] ,
                             [ ak.fill_none(ak.pad_none(events_slice.PFCands.d0, nparticles_per_event,clip=True),-999)           ] ,
                             [ ak.fill_none(ak.pad_none(events_slice.PFCands.dz, nparticles_per_event,clip=True),-999)           ] ,
                             [ ak.fill_none(ak.pad_none(events_slice.PFCands.mass, nparticles_per_event,clip=True),-999)         ] ,
                             [ ak.fill_none(ak.pad_none(events_slice.PFCands.puppiWeight, nparticles_per_event,clip=True),-999)  ] ,
                             [ ak.fill_none(ak.pad_none(events_slice.PFCands.pdgId, nparticles_per_event,clip=True),-999)        ] ,
                             [ ak.fill_none(ak.pad_none(events_slice.PFCands.charge, nparticles_per_event,clip=True),-999)        ] ,
                             [ ak.fill_none(ak.pad_none(events_slice.PFCands.fromPV, nparticles_per_event,clip=True),-999)        ] ,
                             [ ak.fill_none(ak.pad_none(events_slice.PFCands.pvRef, nparticles_per_event,clip=True),-999)         ] ,
                             [ ak.fill_none(ak.pad_none(events_slice.PFCands.pvAssocQuality, nparticles_per_event,clip=True),-999)] ,
                ])
                npz_file=os.environ['PWD']+'/raw/'+dataset+'_file'+str(currentfile)+'_slice_'+str(i)+'_nevent_'+str(len(events_slice))
                np.savez(npz_file,x=particle_list,y=met_list) 
                toc=time.time()
                print('time:',toc-tic)
            currentfile+=1
            if currentfile>=options.endfile:
                print('=================> finished ')
                exit()


