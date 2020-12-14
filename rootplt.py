import ROOT as r
import sys
import datetime
import subprocess
import os
import copy
import collections
import math
import tdr_style.tdrstyle as tdrstyle
import tdr_style.CMS_lumi as CMS_lumi
from array import array
import numpy as np
from utils import load


#set the tdr style
tdrstyle.setTDRStyle()

#change the CMS_lumi variables (see CMS_lumi.py)
CMS_lumi.lumi_7TeV = "4.8 fb^{-1}"
CMS_lumi.lumi_8TeV = "18.3 fb^{-1}"
CMS_lumi.writeExtraText = 1
CMS_lumi.extraText = "Preliminary"
CMS_lumi.lumi_sqrtS = "" # used with iPeriod = 0, e.g. for simulation-only plots (default is an empty string)

iPos = 0
#iPos = 11
if( iPos==0 ): CMS_lumi.relPosX = 0.12
iPeriod = 0


a=load('weight.plt')
color = [1,2,3,4,6,7,30]
label = [
    'bin_edges',
    'weight_pt_hist',
    'weight_eta_hist',
    'weight_puppi_hist',
]
pdg = ['HF Candidate','Electron','Muon','Gamma','Neutral Hadron','Charged Hadron',]
ppdg = ['HF Candidate','Gamma','Neutral Hadron',]
bin_label=['Pt','eta','Puppi']


############################### weight vs. Pt ############################
cpt = r.TCanvas( 'cpt', 'GraphMet Weight', 200, 10, 700, 500 )
cpt.SetLogx()
npt = len(a['weight_pt_hist']['HF Candidate'])
xpt = array( 'd' ) 
ypt = (array( 'd' ), array( 'd' ), array( 'd' ), array( 'd' ), array( 'd' ), array( 'd' ))

for i in range( npt ):
    xpt.append( (a['bin_edges']['Pt'][i]+a['bin_edges']['Pt'][i+1])/2.0 )
    for j in range(len(pdg)):
        ypt[j].append( a['weight_pt_hist'][pdg[j]][i] )
gr={}
for i in range( len(pdg) ):
    gr[i] = r.TGraph( npt, xpt, ypt[i] )
    gr[i].SetLineColor( color[i] )
    gr[i].SetLineWidth( 2 )
    gr[i].SetMarkerColor( color[i] )
    gr[i].SetMarkerStyle( 20 )
    gr[i].SetTitle( 'a simple graph' )
    gr[i].GetXaxis().SetTitle( 'PF P_{T} [GeV]' )
    gr[i].GetYaxis().SetTitle( 'GraphMet Weight' )
    gr[i].Draw( 'AP' )

#CMS_lumi.CMS_lumi(cpt, iPeriod, iPos)
gmul = r.TMultiGraph()
for i in range( len(pdg) ):
    gmul.Add(gr[i])
gmul.SetTitle( '' )
gmul.GetXaxis().SetTitle( 'PF P_{T} [GeV]' )
gmul.GetYaxis().SetTitle( 'GraphMet Weight' )
gmul.GetXaxis().SetRangeUser(0.01,30)
gmul.GetYaxis().SetRangeUser(0,1.4)
gmul.Draw( 'AP' )
cpt.Update()
legend_pt = r.TLegend(0.2,0.6,0.4,0.95)
legend_pt.SetFillStyle(0)
legend_pt.SetBorderSize(0)
legend_pt.SetTextSize(0.04)
legend_pt.SetTextFont(42)
for i in range( len(pdg) ):
    legend_pt.AddEntry(gr[i], pdg[i], "PE")
legend_pt.Draw("same")
cpt.Modified()
cpt.Update()
cpt.SetFillColor(0)
cpt.SetBorderMode(0)
cpt.SetBorderSize(2)
cpt.SetFrameBorderMode(0)

############################### weight vs. eta ############################
ceta = r.TCanvas( 'ceta', 'GraphMet Weight', 200, 10, 700, 500 )
neta = len(a['weight_eta_hist']['HF Candidate'])
xeta = array( 'd' )
yeta = (array( 'd' ), array( 'd' ), array( 'd' ), array( 'd' ), array( 'd' ), array( 'd' ))

for i in range( neta ):
    xeta.append( (a['bin_edges']['eta'][i]+a['bin_edges']['eta'][i+1])/2.0 )
    for j in range(len(pdg)):
        yeta[j].append( a['weight_eta_hist'][pdg[j]][i] )
greta={}
for i in range( len(pdg) ):
    greta[i] = r.TGraph( neta, xeta, yeta[i] )
    greta[i].SetLineColor( color[i] )
    greta[i].SetLineWidth( 2 )
    greta[i].SetMarkerColor( color[i] )
    greta[i].SetMarkerStyle( 20 )
    greta[i].SetTitle( 'a simple graph' )
    greta[i].GetXaxis().SetTitle( 'PF |#eta|' )
    greta[i].GetYaxis().SetTitle( 'GraphMet Weight' )
    greta[i].Draw( 'AP' )
gmuleta = r.TMultiGraph()
for i in range( len(pdg) ):
    gmuleta.Add(greta[i])
gmuleta.GetXaxis().SetTitle( 'PF |#eta|' )
gmuleta.GetYaxis().SetTitle( 'GraphMet Weight' )
gmuleta.GetXaxis().SetRangeUser(0,5)
gmuleta.GetYaxis().SetRangeUser(0,1.4)
gmuleta.Draw( 'AP' )
ceta.Update()
legend_eta = r.TLegend(0.65,0.6,0.98,0.95)
legend_eta.SetFillStyle(0)
legend_eta.SetBorderSize(0)
legend_eta.SetTextSize(0.04)
legend_eta.SetTextFont(42)
for i in range( len(pdg) ):
    legend_eta.AddEntry(greta[i], pdg[i], "PE")
legend_eta.Draw("same")
ceta.Modified()
ceta.Update()
ceta.SetFillColor(0)
ceta.SetBorderMode(0)
ceta.SetBorderSize(2)
ceta.SetFrameBorderMode(0)





############################### weight vs. puppi ############################
cpp = r.TCanvas( 'cpp', 'GraphMet Weight', 200, 10, 700, 500 )
npp = len(a['weight_puppi_hist']['HF Candidate'])
xpp = array( 'd' )
ypp = (array( 'd' ), array( 'd' ), array( 'd' ))

for i in range( npp ):
    xpp.append( (a['bin_edges']['Puppi'][i]+a['bin_edges']['Puppi'][i+1])/2.0 )
    for j in range(len(ppdg)):
        ypp[j].append( a['weight_puppi_hist'][ppdg[j]][i] )
grpp={}
for i in range( len(ppdg) ):
    grpp[i] = r.TGraph( npp, xpp, ypp[i] )
    grpp[i].SetLineColor( color[i] )
    grpp[i].SetLineWidth( 2 )
    grpp[i].SetMarkerColor( color[i] )
    grpp[i].SetMarkerStyle( 20 )
    grpp[i].SetTitle( 'a simple graph' )
    grpp[i].GetXaxis().SetTitle( 'PUPPI Weight' )
    grpp[i].GetYaxis().SetTitle( 'GraphMet Weight' )
    grpp[i].Draw( 'AP' )

gmulpp = r.TMultiGraph()
for i in range( len(ppdg) ):
    gmulpp.Add(grpp[i])
#gmulpp.SetTitle( 'a simple graph' )
gmulpp.GetXaxis().SetTitle( 'PUPPI Weight' )
gmulpp.GetYaxis().SetTitle( 'GraphMet Weight' )
gmulpp.GetXaxis().SetRangeUser(0,1)
gmulpp.GetYaxis().SetRangeUser(0,1.4)
gmulpp.Draw( 'AP' )

cpp.Update()
legend_pp = r.TLegend(0.55,0.8,0.98,0.95)
legend_pp.SetFillStyle(0)
legend_pp.SetBorderSize(0)
legend_pp.SetTextSize(0.04)
legend_pp.SetTextFont(42)
for i in range( len(ppdg) ):
    legend_pp.AddEntry(grpp[i], ppdg[i], "PE")
legend_pp.Draw("same")
cpp.Modified()
cpp.Update()
cpp.SetFillColor(0)
cpp.SetBorderMode(0)
cpp.SetBorderSize(2)
cpp.SetFrameBorderMode(0)


(input("Please enter an integer to exit: "))











