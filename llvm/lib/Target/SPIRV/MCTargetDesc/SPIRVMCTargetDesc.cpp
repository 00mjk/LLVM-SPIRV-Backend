//===-- SPIRVMCTargetDesc.cpp - SPIR-V Target Descriptions ----*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides SPIR-V specific target descriptions.
//
//===----------------------------------------------------------------------===//

#include "SPIRVMCTargetDesc.h"
#include "SPIRVInstPrinter.h"
#include "SPIRVMCAsmInfo.h"
#include "SPIRVTargetStreamer.h"
#include "TargetInfo/SPIRVTargetInfo.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/TargetRegistry.h"

#define GET_INSTRINFO_MC_DESC
#include "SPIRVGenInstrInfo.inc"

#define GET_SUBTARGETINFO_MC_DESC
#include "SPIRVGenSubtargetInfo.inc"

#define GET_REGINFO_MC_DESC
#include "SPIRVGenRegisterInfo.inc"

using namespace llvm;

static MCInstrInfo *createSPIRVMCInstrInfo() {
  MCInstrInfo *X = new MCInstrInfo();
  InitSPIRVMCInstrInfo(X);
  return X;
}

static MCRegisterInfo *createSPIRVMCRegisterInfo(const Triple &TT) {
  MCRegisterInfo *X = new MCRegisterInfo();
  return X;
}

static MCSubtargetInfo *
createSPIRVMCSubtargetInfo(const Triple &TT, StringRef CPU, StringRef FS) {
  return createSPIRVMCSubtargetInfoImpl(TT, CPU, /*TuneCPU*/ CPU, FS);
}

static MCStreamer *
createSPIRVMCStreamer(const Triple &T, MCContext &Ctx,
                      std::unique_ptr<MCAsmBackend> &&MAB,
                      std::unique_ptr<MCObjectWriter> &&OW,
                      std::unique_ptr<MCCodeEmitter> &&Emitter, bool RelaxAll) {
  return createSPIRVStreamer(Ctx, std::move(MAB), std::move(OW),
                             std::move(Emitter), RelaxAll);
}

static MCTargetStreamer *createTargetAsmStreamer(MCStreamer &S,
                                                 formatted_raw_ostream &,
                                                 MCInstPrinter *, bool) {
  return new SPIRVTargetStreamer(S);
}

static MCInstPrinter *createSPIRVMCInstPrinter(const Triple &T,
                                               unsigned SyntaxVariant,
                                               const MCAsmInfo &MAI,
                                               const MCInstrInfo &MII,
                                               const MCRegisterInfo &MRI) {
  assert(SyntaxVariant == 0);
  return new SPIRVInstPrinter(MAI, MII, MRI);
}

namespace {

class SPIRVMCInstrAnalysis : public MCInstrAnalysis {
public:
  explicit SPIRVMCInstrAnalysis(const MCInstrInfo *Info)
      : MCInstrAnalysis(Info) {}
};

} // end anonymous namespace

static MCInstrAnalysis *createSPIRVInstrAnalysis(const MCInstrInfo *Info) {
  return new SPIRVMCInstrAnalysis(Info);
}

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeSPIRVTargetMC() {
  for (Target *T : {&getTheSPIRV32Target(), &getTheSPIRV64Target(),
                    &getTheSPIRVLogicalTarget()}) {
    RegisterMCAsmInfo<SPIRVMCAsmInfo> X(*T);
    TargetRegistry::RegisterMCInstrInfo(*T, createSPIRVMCInstrInfo);
    TargetRegistry::RegisterMCRegInfo(*T, createSPIRVMCRegisterInfo);
    TargetRegistry::RegisterMCSubtargetInfo(*T, createSPIRVMCSubtargetInfo);
    TargetRegistry::RegisterSPIRVStreamer(*T, createSPIRVMCStreamer);
    TargetRegistry::RegisterMCInstPrinter(*T, createSPIRVMCInstPrinter);
    TargetRegistry::RegisterMCInstrAnalysis(*T, createSPIRVInstrAnalysis);
    TargetRegistry::RegisterMCCodeEmitter(*T, createSPIRVMCCodeEmitter);
    TargetRegistry::RegisterMCAsmBackend(*T, createSPIRVAsmBackend);
    TargetRegistry::RegisterAsmTargetStreamer(*T, createTargetAsmStreamer);
  }
}
