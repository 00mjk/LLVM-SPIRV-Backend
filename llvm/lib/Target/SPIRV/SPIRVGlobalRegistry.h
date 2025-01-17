//===-- SPIRVGlobalRegistry.h - SPIR-V Global Registry ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SPIRVGlobalRegistry is used to maintain rich type information required for
// SPIR-V even after lowering from LLVM IR to GMIR. It can convert an llvm::Type
// into an OpTypeXXX instruction, and map it to a virtual register. Also it
// builds and supports consistency of constants and global variables.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SPIRV_SPIRVTYPEMANAGER_H
#define LLVM_LIB_TARGET_SPIRV_SPIRVTYPEMANAGER_H

#include "MCTargetDesc/SPIRVBaseInfo.h"
#include "SPIRVDuplicatesTracker.h"
#include "SPIRVInstrInfo.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"

namespace AQ = AccessQualifier;

namespace llvm {
using SPIRVType = const MachineInstr;

class SPIRVGlobalRegistry {
  // Registers holding values which have types associated with them.
  // Initialized upon VReg definition in IRTranslator.
  // Do not confuse this with DuplicatesTracker as DT maps Type* to <MF, Reg>
  // where Reg = OpType...
  // while VRegToTypeMap tracks SPIR-V type assigned to other regs (i.e. not
  // type-declaring ones).
  DenseMap<MachineFunction *, DenseMap<Register, SPIRVType *>> VRegToTypeMap;

  SPIRVGeneralDuplicatesTracker DT;

  DenseMap<SPIRVType *, const Type *> SPIRVToLLVMType;

  // Look for an equivalent of the newType in the map. Return the equivalent
  // if it's found, otherwise insert newType to the map and return the type.
  const MachineInstr *checkSpecialInstr(const SPIRV::SpecialTypeDescriptor &TD,
                                        MachineIRBuilder &MIRBuilder);

  SmallPtrSet<const Type *, 4> TypesInProcessing;
  DenseMap<const Type *, SPIRVType *> ForwardPointerTypes;

  // Number of bits pointers and size_t integers require.
  const unsigned PointerSize;

  // Add a new OpTypeXXX instruction without checking for duplicates.
  SPIRVType *createSPIRVType(const Type *Type, MachineIRBuilder &MIRBuilder,
                             AQ::AccessQualifier accessQual = AQ::ReadWrite,
                             bool EmitIR = true);
  SPIRVType *findSPIRVType(const Type *Ty, MachineIRBuilder &MIRBuilder,
                           AQ::AccessQualifier accessQual = AQ::ReadWrite,
                           bool EmitIR = true);
  SPIRVType *restOfCreateSPIRVType(const Type *Type,
                                   MachineIRBuilder &MIRBuilder,
                                   AQ::AccessQualifier AccessQual, bool EmitIR);

public:
  void add(const Constant *C, MachineFunction *MF, Register R) {
    DT.add(C, MF, R);
  }

  void add(const GlobalVariable *GV, MachineFunction *MF, Register R) {
    DT.add(GV, MF, R);
  }

  void add(const Function *F, MachineFunction *MF, Register R) {
    DT.add(F, MF, R);
  }

  void add(const Argument *Arg, MachineFunction *MF, Register R) {
    DT.add(Arg, MF, R);
  }

  bool find(const Constant *C, MachineFunction *MF, Register &R) {
    return DT.find(C, MF, R);
  }

  bool find(const GlobalVariable *GV, MachineFunction *MF, Register &R) {
    return DT.find(GV, MF, R);
  }

  bool find(const Function *F, MachineFunction *MF, Register &R) {
    return DT.find(F, MF, R);
  }

  const typename SPIRVDuplicatesTracker<Function>::StorageTy &getFuncAllUses() {
    return DT.getFuncs()->getAllUses();
  }

  void buildDepsGraph(std::vector<SPIRV::DTSortableEntry *> &Graph,
                      MachineModuleInfo *MMI = nullptr) {
    DT.buildDepsGraph(Graph, MMI);
  }

  // This interface is for walking the map in GlobalTypesAndRegNumPass.
  // SpecialInstrMapTy &getSpecialTypesAndConstsMap() {
  //   return SpecialTypesAndConstsMap;
  // }

  SPIRVGlobalRegistry(unsigned PointerSize);

  MachineFunction *CurMF;

  // Get or create a SPIR-V type corresponding the given LLVM IR type,
  // and map it to the given VReg by creating an ASSIGN_TYPE instruction.
  SPIRVType *assignTypeToVReg(const Type *Type, Register VReg,
                              MachineIRBuilder &MIRBuilder,
                              AQ::AccessQualifier AccessQual = AQ::ReadWrite,
                              bool EmitIR = true);
  SPIRVType *assignIntTypeToVReg(unsigned BitWidth, Register VReg,
                                 MachineInstr &I, const SPIRVInstrInfo &TII);
  SPIRVType *assignVectTypeToVReg(SPIRVType *BaseType, unsigned NumElements,
                                  Register VReg, MachineInstr &I,
                                  const SPIRVInstrInfo &TII);

  // In cases where the SPIR-V type is already known, this function can be
  // used to map it to the given VReg via an ASSIGN_TYPE instruction.
  void assignSPIRVTypeToVReg(SPIRVType *Type, Register VReg,
                             MachineFunction &MF);

  // Either generate a new OpTypeXXX instruction or return an existing one
  // corresponding to the given LLVM IR type.
  // EmitIR controls if we emit GMIR or SPV constants (e.g. for array sizes)
  // because this method may be called from InstructionSelector and we don't
  // want to emit extra IR instructions there.
  SPIRVType *
  getOrCreateSPIRVType(const Type *Type, MachineIRBuilder &MIRBuilder,
                       AQ::AccessQualifier accessQual = AQ::ReadWrite,
                       bool EmitIR = true);

  const Type *getTypeForSPIRVType(const SPIRVType *Ty) const {
    auto Res = SPIRVToLLVMType.find(Ty);
    assert(Res != SPIRVToLLVMType.end());
    return Res->second;
  }

  // Either generate a new OpTypeXXX instruction or return an existing one
  // corresponding to the given string containing the name of the builtin type.
  SPIRVType *getOrCreateSPIRVTypeByName(StringRef TypeStr,
                                        MachineIRBuilder &MIRBuilder);

  // Return the SPIR-V type instruction corresponding to the given VReg, or
  // nullptr if no such type instruction exists.
  SPIRVType *getSPIRVTypeForVReg(Register VReg) const;

  // Whether the given VReg has a SPIR-V type mapped to it yet.
  bool hasSPIRVTypeForVReg(Register VReg) const {
    return getSPIRVTypeForVReg(VReg) != nullptr;
  }

  // Return the VReg holding the result of the given OpTypeXXX instruction.
  Register getSPIRVTypeID(const SPIRVType *SpirvType) const;

  void setCurrentFunc(MachineFunction &MF) { CurMF = &MF; }

  // Whether the given VReg has an OpTypeXXX instruction mapped to it with the
  // given opcode (e.g. OpTypeFloat).
  bool isScalarOfType(Register VReg, unsigned TypeOpcode) const;

  // Return true if the given VReg's assigned SPIR-V type is either a scalar
  // matching the given opcode, or a vector with an element type matching that
  // opcode (e.g. OpTypeBool, or OpTypeVector %x 4, where %x is OpTypeBool).
  bool isScalarOrVectorOfType(Register VReg, unsigned TypeOpcode) const;

  // For vectors or scalars of ints/floats, return the scalar type's bitwidth.
  unsigned getScalarOrVectorBitWidth(const SPIRVType *Type) const;

  // For integer vectors or scalars, return whether the integers are signed.
  bool isScalarOrVectorSigned(const SPIRVType *Type) const;

  // Gets the storage class of the pointer type assigned to this vreg.
  StorageClass::StorageClass getPointerStorageClass(Register VReg) const;

  // Return the number of bits SPIR-V pointers and size_t variables require.
  unsigned getPointerSize() const { return PointerSize; }

private:
  // Internal methods for creating types which are unsafe in duplications
  // check sense hence can only be called within getOrCreateSPIRVType callstack
  SPIRVType *getOpTypeBool(MachineIRBuilder &MIRBuilder);

  SPIRVType *getOpTypeInt(uint32_t Width, MachineIRBuilder &MIRBuilder,
                          bool IsSigned = false);

  SPIRVType *getOpTypeFloat(uint32_t Width, MachineIRBuilder &MIRBuilder);

  SPIRVType *getOpTypeVoid(MachineIRBuilder &MIRBuilder);

  SPIRVType *getOpTypeVector(uint32_t NumElems, SPIRVType *ElemType,
                             MachineIRBuilder &MIRBuilder);

  SPIRVType *getOpTypeArray(uint32_t NumElems, SPIRVType *ElemType,
                            MachineIRBuilder &MIRBuilder, bool EmitIR = true);

  SPIRVType *getOpTypeOpaque(const StructType *Ty,
                             MachineIRBuilder &MIRBuilder);

  SPIRVType *getOpTypeStruct(const StructType *Ty, MachineIRBuilder &MIRBuilder,
                             bool EmitIR = true);

  SPIRVType *getOpTypePointer(StorageClass::StorageClass SC,
                              SPIRVType *ElemType, MachineIRBuilder &MIRBuilder,
                              Register Reg);

  SPIRVType *getOpTypeForwardPointer(StorageClass::StorageClass SC,
                                     MachineIRBuilder &MIRBuilder);

  SPIRVType *getOpTypeFunction(SPIRVType *RetType,
                               const SmallVectorImpl<SPIRVType *> &ArgTypes,
                               MachineIRBuilder &MIRBuilder);

  SPIRVType *getOpTypeByOpcode(MachineIRBuilder &MIRBuilder, unsigned Opcode);

  SPIRVType *handleOpenCLBuiltin(const StructType *Ty,
                                 MachineIRBuilder &MIRBuilder,
                                 AQ::AccessQualifier AccQual);

  SPIRVType *
  getOrCreateOpenCLOpaqueType(const StructType *Ty,
                              MachineIRBuilder &MIRBuilder,
                              AQ::AccessQualifier AccQual = AQ::ReadWrite);

  SPIRVType *handleSPIRVBuiltin(const StructType *Ty,
                                MachineIRBuilder &MIRBuilder,
                                AQ::AccessQualifier AccQual);

  SPIRVType *
  getOrCreateSPIRVOpaqueType(const StructType *Ty, MachineIRBuilder &MIRBuilder,
                             AQ::AccessQualifier AccQual = AQ::ReadWrite);

  std::tuple<Register, ConstantInt *, bool> getOrCreateConstIntReg(
      uint64_t Val, SPIRVType *SpvType, MachineIRBuilder *MIRBuilder,
      MachineInstr *I = nullptr, const SPIRVInstrInfo *TII = nullptr);
  SPIRVType *restOfCreateSPIRVType(Type *LLVMTy, SPIRVType *SpirvType);

public:
  Register buildConstantInt(uint64_t Val, MachineIRBuilder &MIRBuilder,
                            SPIRVType *SpvType = nullptr, bool EmitIR = true);
  Register getOrCreateConstInt(uint64_t Val, MachineInstr &I,
                               SPIRVType *SpvType, const SPIRVInstrInfo &TII);
  Register buildConstantFP(APFloat Val, MachineIRBuilder &MIRBuilder,
                           SPIRVType *SpvType = nullptr);
  Register buildConstantIntVector(uint64_t Val, MachineIRBuilder &MIRBuilder,
                                  SPIRVType *SpvType, bool EmitIR = true);
  Register getOrCreateConsIntVector(uint64_t Val, MachineInstr &I,
                                    SPIRVType *SpvType,
                                    const SPIRVInstrInfo &TII);
  Register buildConstantSampler(Register Res, unsigned AddrMode, unsigned Param,
                                unsigned FilerMode,
                                MachineIRBuilder &MIRBuilder,
                                SPIRVType *SpvType);
  Register getOrCreateUndef(MachineInstr &I, SPIRVType *SpvType,
                            const SPIRVInstrInfo &TII);
  Register
  buildGlobalVariable(Register Reg, SPIRVType *BaseType, StringRef Name,
                      const GlobalValue *GV, StorageClass::StorageClass Storage,
                      const MachineInstr *Init, bool IsConst, bool HasLinkageTy,
                      LinkageType::LinkageType LinkageType,
                      MachineIRBuilder &MIRBuilder, bool IsInstSelector);

  // Convenient helpers for getting types with check for duplicates.
  SPIRVType *getOrCreateSPIRVIntegerType(unsigned BitWidth,
                                         MachineIRBuilder &MIRBuilder);
  SPIRVType *getOrCreateSPIRVIntegerType(unsigned BitWidth, MachineInstr &I,
                                         const SPIRVInstrInfo &TII);
  SPIRVType *getOrCreateSPIRVBoolType(MachineIRBuilder &MIRBuilder);
  SPIRVType *getOrCreateSPIRVBoolType(MachineInstr &I,
                                      const SPIRVInstrInfo &TII);
  SPIRVType *getOrCreateSPIRVVectorType(SPIRVType *BaseType,
                                        unsigned NumElements,
                                        MachineIRBuilder &MIRBuilder);
  SPIRVType *getOrCreateSPIRVVectorType(SPIRVType *BaseType,
                                        unsigned NumElements, MachineInstr &I,
                                        const SPIRVInstrInfo &TII);
  SPIRVType *getOrCreateSPIRVPointerType(
      SPIRVType *BaseType, MachineIRBuilder &MIRBuilder,
      StorageClass::StorageClass SClass = StorageClass::Function);
  SPIRVType *getOrCreateSPIRVPointerType(
      SPIRVType *BaseType, MachineInstr &I, const SPIRVInstrInfo &TII,
      StorageClass::StorageClass SC = StorageClass::Function);

  SPIRVType *getOrCreateOpTypeImage(MachineIRBuilder &MIRBuilder,
                                    SPIRVType *SampledType, Dim::Dim Dim,
                                    uint32_t Depth, uint32_t Arrayed,
                                    uint32_t Multisampled, uint32_t Sampled,
                                    ImageFormat::ImageFormat ImageFormat,
                                    AQ::AccessQualifier AccQual);

  SPIRVType *getOrCreateOpTypeSampler(MachineIRBuilder &MIRBuilder);

  SPIRVType *getOrCreateOpTypeSampledImage(SPIRVType *ImageType,
                                           MachineIRBuilder &MIRBuilder);

  SPIRVType *getOrCreateOpTypePipe(MachineIRBuilder &MIRBuilder,
                                   AQ::AccessQualifier AccQual);
};
} // end namespace llvm
#endif // LLLVM_LIB_TARGET_SPIRV_SPIRVTYPEMANAGER_H
