import FWCore.ParameterSet.Config as cms
import FWCore.Utilities.FileUtils as FileUtils
from FWCore.ParameterSet.VarParsing import VarParsing
import os


options = VarParsing ('analysis')

options.register( 'executionMode',
                  1,
                  VarParsing.multiplicity.singleton,
                  VarParsing.varType.int,
                  "0 for particle data, 1 for jet data, 2 for jet prediction"
               )
options.parseArguments()
process = cms.Process("AK4jets")



# QG likelihood
process.load("MLJetReconstruction.JetAnalyzer.QGLikelihood_cfi")
process.load("RecoJets.JetProducers.QGTagger_cfi")
process.QGTagger.srcJets = cms.InputTag("slimmedJets")
process.QGTagger.jetsLabel = cms.string("QGL_AK4PFchs")

# File service
process.load("CommonTools.UtilAlgos.TFileService_cfi")
process.TFileService.fileName=cms.string("JetNtuple_RunIISummer16_13TeV_MC.root")

# Load up the filelist
filePath=os.environ["CMSSW_BASE"]+"/src/MLJetReconstruction/JetAnalyzer/python/"
fileList = FileUtils.loadListFromFile(filePath+"filelist.txt")

process.source = cms.Source("PoolSource",
fileNames = cms.untracked.vstring(fileList[0:21])

process.AK4jets = cms.EDAnalyzer("JetAnalyzer",
	## jet, PF and generator level collections ##
	jets = cms.InputTag("slimmedJetsPuppi"),
	pfCands = cms.InputTag("packedPFCandidates"),
	genJets = cms.InputTag("slimmedGenJets"),
	genEventInfo = cms.InputTag("generator"),
	## good primary vertices ##
	vertices = cms.InputTag("offlineSlimmedPrimaryVertices"),
	confGoodVtxNdof = cms.double(4),
	confGoodVtxZ = cms.double(24),
	confGoodVtxRho = cms.double(2),
	## pileup and rhos ##
	pileupInfo = cms.InputTag("slimmedAddPileupInfo"),
	pfRhoAll = cms.InputTag("fixedGridRhoFastjetAll"),
	pfRhoCentral = cms.InputTag("fixedGridRhoFastjetCentral"),
	pfRhoCentralNeutral = cms.InputTag("fixedGridRhoFastjetCentralNeutral"),
	pfRhoCentralChargedPileUp = cms.InputTag("fixedGridRhoFastjetCentralChargedPileUp")
)

# Choose how many events to process (-1 = all)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

# Report execution progress
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

process.p = cms.Path(process.QGTagger + process.AK4jets)
