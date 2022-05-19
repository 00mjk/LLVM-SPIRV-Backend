//===- SPIRVBuiltins.cpp - SPIR-V Built-in Functions ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering builtin function calls and types using their 
// demangled names and TableGen records.
//
//===----------------------------------------------------------------------===//

#include "SPIRVBuiltins.h"
#include "SPIRV.h"
#include "SPIRVUtils.h"

#include "llvm/IR/IntrinsicsSPIRV.h"

#include <sstream>
#include <string>
#include <tuple>
#include <vector>

using namespace llvm;
using namespace SPIRV;

#define DEBUG_TYPE "spirv-builtins"

namespace {
#define GET_InstructionSet_DECL
#define GET_BuiltinGroup_DECL
#include "SPIRVGenTables.inc"

// Struct holding a demangled builtin record.
struct DemangledBuiltin {
  StringRef Name;
  InstructionSet Set;
  BuiltinGroup Group;
  int8_t MinNumArgs;
  int8_t MaxNumArgs;
};

#define GET_DemangledBuiltins_DECL
#define GET_DemangledBuiltins_IMPL

struct IncomingCall {
  const std::string BuiltinName;
  const DemangledBuiltin *Builtin;

  const Register ReturnRegister;
  const SPIRVType *ReturnType;
  const SmallVectorImpl<Register> &Arguments;

  IncomingCall(const std::string BuiltinName, const DemangledBuiltin *Builtin,
               const Register ReturnRegister, const SPIRVType *ReturnType,
               const SmallVectorImpl<Register> &Arguments)
      : BuiltinName(BuiltinName), Builtin(Builtin),
        ReturnRegister(ReturnRegister), ReturnType(ReturnType),
        Arguments(Arguments) {}
};

struct ExtendedBuiltin {
  StringRef Name;
  InstructionSet Set;
  uint32_t Number;
};

#define GET_ExtendedBuiltins_DECL
#define GET_ExtendedBuiltins_IMPL

struct NativeBuiltin {
  StringRef Name;
  InstructionSet Set;
  uint32_t Opcode;
};

#define GET_NativeBuiltins_DECL
#define GET_NativeBuiltins_IMPL

struct GroupBuiltin {
  StringRef Name;
  uint32_t Opcode;
  uint32_t GroupOperation;
  bool IsElect;
  bool IsAllOrAny;
  bool IsAllEqual;
  bool IsBallot;
  bool IsInverseBallot;
  bool IsBallotBitExtract;
  bool IsBallotFindBit;
  bool IsLogical;
  bool NoGroupOperation;
  bool HasBoolArg;
};

#define GET_GroupBuiltins_DECL
#define GET_GroupBuiltins_IMPL

struct BuiltinVariable {
  StringRef Name;
  InstructionSet Set;
  ::BuiltIn::BuiltIn Value;
};

using namespace BuiltIn;
#define GET_BuiltinVariables_DECL
#define GET_BuiltinVariables_IMPL

#include "SPIRVGenTables.inc"

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Misc functions for looking up builtins and veryfying requirements using
// TableGen records
//===----------------------------------------------------------------------===//

/// Looks up the demangled builtin call in the SPIRVBuiltins.td records using
/// the provided \p DemangledCall and specified \p Set.
///
/// The lookup follows the following algorithm, returning the first successful
/// match:
/// 1. Search with the plain demangled name (expecting a 1:1 match).
/// 2. Search with the demangled name + prefix signyfying the type of the first
/// argument.
/// 3. Remove the suffix from the demangled name and repeat the search.
///
/// \returns Wrapper around the demangled call and found builtin definition.
static std::unique_ptr<const IncomingCall>
lookupBuiltin(StringRef DemangledCall,
              ExternalInstructionSet::InstructionSet Set,
              Register ReturnRegister, const SPIRVType *ReturnType,
              const SmallVectorImpl<Register> &Arguments) {
  // Extract the builtin function name and types of arguments from the call
  // skeleton.
  std::string BuiltinName =
      DemangledCall.substr(0, DemangledCall.find('(')).str();
  std::vector<std::string> BuiltinArgumentTypes;
  {
    std::stringstream Stream(
        DemangledCall.substr(DemangledCall.find('(') + 1).drop_back(1).str());
    std::string DemangledArgument;
    while (std::getline(Stream, DemangledArgument, ',')) {
      BuiltinArgumentTypes.push_back(StringRef(DemangledArgument).trim().str());
    }
  }

  // Look up the builtin in the defined set. Start with the plain demangled 
  // name, expecting a 1:1 match in the defined builtin set.
  if (const DemangledBuiltin *Builtin = lookupBuiltin(BuiltinName, Set))
    return std::make_unique<IncomingCall>(BuiltinName, Builtin, ReturnRegister,
                                          ReturnType, Arguments);

  // If the initial look up was unsuccessful and the demangled call takes at
  // least 1 argument, add a prefix signifying the type of the first argument
  // and repeat the search.
  if (BuiltinArgumentTypes.size() >= 1) {
    char FirstArgumentType = BuiltinArgumentTypes[0][0];
    // Prefix to be added to the builtin's name for lookup.
    // For example, OpenCL "abs" taking an unsigned value has a prefix "u_".
    std::string Prefix;

    switch (FirstArgumentType) {
    // Unsigned
    case 'u':
      if (Set == ExternalInstructionSet::OpenCL_std)
        Prefix = "u_";
      else if (Set == ExternalInstructionSet::GLSL_std_450)
        Prefix = "u";
      break;
    // Signed
    case 'c':
    case 's':
    case 'i':
    case 'l':
      if (Set == ExternalInstructionSet::OpenCL_std)
        Prefix = "s_";
      else if (Set == ExternalInstructionSet::GLSL_std_450)
        Prefix = "s";
      break;
    // Floating-point
    case 'f':
    case 'd':
    case 'h':
      if (Set == ExternalInstructionSet::OpenCL_std ||
          Set == ExternalInstructionSet::GLSL_std_450)
        Prefix = "f";
      break;
    }

    // If argument-type name prefix was added, look up the builtin again.
    const DemangledBuiltin *Builtin;
    if (!Prefix.empty() &&
        (Builtin = lookupBuiltin(Prefix + BuiltinName, Set))) {
      return std::make_unique<IncomingCall>(
          BuiltinName, Builtin, ReturnRegister, ReturnType, Arguments);
    }
  }

  // If the lookups above were unsuccessful, try removing the suffix if present.
  // Some OpenCL functions have suffixes at the end of the demangled name. For
  // example, the "convert" builtin has a destination type and rounding mode
  // specified in the name:
  //
  // destType *convert_destType<_sat><_roundingMode>*(sourceType)
  //
  // Do the search without the <_suffix>*, but keep it in the returned struct
  // for later use in the implementation code.
  if (BuiltinName.find('_') != std::string::npos) {
    // Try removing up to 3 suffixes until a builtin is found in the set.
    std::string Subname = BuiltinName;
    for (unsigned i = 0; i < 3; i++) {
      Subname = Subname.substr(0, Subname.find_last_of('_'));

      if (const DemangledBuiltin *Builtin = lookupBuiltin(Subname, Set)) {
        if (Builtin->Group == Convert)
          return std::make_unique<IncomingCall>(
              BuiltinName, Builtin, ReturnRegister, ReturnType, Arguments);
      }
    }
  }

  // No builtin with such name was found in the set.
  return nullptr;
}

/// Verify if the provided \p Arguments meet the requirments of the given \p
/// Builtin.
static bool verifyBuiltinArgs(const DemangledBuiltin *Builtin,
                              const SmallVectorImpl<Register> &Arguments) {
  assert(Arguments.size() >= Builtin->MinNumArgs &&
         "Too little arguments to generate the builtin");
  assert(Arguments.size() <= Builtin->MaxNumArgs &&
         "Too many arguments to generate the builtin");

  return true;
}

//===----------------------------------------------------------------------===//
// Helper functions for building misc instructions
//===----------------------------------------------------------------------===//

/// Helper function building either a resulting scalar or vector bool register
/// depending on the expected \p ResultType.
///
/// \returns Tuple of the resulting register and its type.
static std::tuple<Register, SPIRVType *>
buildBoolRegister(MachineIRBuilder &MIRBuilder, const SPIRVType *ResultType,
                  SPIRVGlobalRegistry *GR) {
  LLT Type;
  SPIRVType *BoolType = GR->getOrCreateSPIRVBoolType(MIRBuilder);

  if (ResultType->getOpcode() == OpTypeVector) {
    unsigned VectorElements = ResultType->getOperand(2).getImm();
    BoolType =
        GR->getOrCreateSPIRVVectorType(BoolType, VectorElements, MIRBuilder);
    auto LLVMVectorType =
        cast<FixedVectorType>(GR->getTypeForSPIRVType(BoolType));
    Type = LLT::vector(LLVMVectorType->getElementCount(), 1);
  } else {
    Type = LLT::scalar(1);
  }

  Register ResultRegister =
      MIRBuilder.getMRI()->createGenericVirtualRegister(Type);
  GR->assignSPIRVTypeToVReg(BoolType, ResultRegister, MIRBuilder.getMF());
  return std::make_tuple(ResultRegister, BoolType);
}

/// Helper function for building either a vector or scalar select instruction
/// depending on the expected \p ResultType.
static bool buildSelectInst(MachineIRBuilder &MIRBuilder,
                            Register ReturnRegister, Register SourceRegister,
                            const SPIRVType *ReturnType,
                            SPIRVGlobalRegistry *GR) {
  Register TrueConst;
  Register FalseConst;
  unsigned Bits = GR->getScalarOrVectorBitWidth(ReturnType);

  if (ReturnType->getOpcode() == OpTypeVector) {
    auto AllOnes = APInt::getAllOnesValue(Bits).getZExtValue();
    TrueConst = GR->buildConstantIntVector(AllOnes, MIRBuilder, ReturnType);
    FalseConst = GR->buildConstantIntVector(0, MIRBuilder, ReturnType);
  } else {
    TrueConst = GR->buildConstantInt(1, MIRBuilder, ReturnType);
    FalseConst = GR->buildConstantInt(0, MIRBuilder, ReturnType);
  }

  return MIRBuilder.buildSelect(ReturnRegister, SourceRegister, TrueConst,
                                FalseConst);
}

/// Helper function for building a load instruction loading into the 
/// \p DestinationReg.
static Register buildLoadInst(SPIRVType *BaseType, Register PtrRegister,
                          MachineIRBuilder &MIRBuilder, SPIRVGlobalRegistry *GR,
                          LLT LowLevelType, Register DestinationReg = Register(0)) {
  const auto MRI = MIRBuilder.getMRI();
  if (!DestinationReg.isValid()) {
    DestinationReg = MRI->createVirtualRegister(&SPIRV::IDRegClass);
    MRI->setType(DestinationReg, LLT::scalar(32));
    GR->assignSPIRVTypeToVReg(BaseType, DestinationReg, MIRBuilder.getMF());
  }

  // TODO: consider using correct address space and alignment
  // p0 is canonical type for selection though
  MachinePointerInfo PtrInfo = MachinePointerInfo();
  MIRBuilder.buildLoad(DestinationReg, PtrRegister, PtrInfo, Align());
  return DestinationReg;
}

/// Helper function for building a load instruction for loading a builtin global
/// variable of \p BuiltinValue value.
static Register buildBuiltinVariableLoad(MachineIRBuilder &MIRBuilder,
                                         SPIRVType *VariableType,
                                         SPIRVGlobalRegistry *GR,
                                         ::BuiltIn::BuiltIn BuiltinValue,
                                         LLT LLType,
                                         Register Reg = Register(0)) {
  Register NewRegister =
      MIRBuilder.getMRI()->createVirtualRegister(&SPIRV::IDRegClass);
  MIRBuilder.getMRI()->setType(NewRegister,
                               LLT::pointer(0, GR->getPointerSize()));
  SPIRVType *PtrType = GR->getOrCreateSPIRVPointerType(VariableType, MIRBuilder,
                                                       StorageClass::Input);
  GR->assignSPIRVTypeToVReg(PtrType, NewRegister, MIRBuilder.getMF());

  // Set up the global OpVariable with the necessary builtin decorations
  Register Variable = GR->buildGlobalVariable(
      NewRegister, PtrType, getLinkStringForBuiltIn(BuiltinValue), nullptr,
      StorageClass::Input, nullptr, true, true, LinkageType::Import, MIRBuilder,
      false);

  // Load the value from the global variable
  Register LoadedRegister =
      buildLoadInst(VariableType, Variable, MIRBuilder, GR, LLType, Reg);
  MIRBuilder.getMRI()->setType(LoadedRegister, LLType);
  return LoadedRegister;
}

/// Helper external function for inserting ASSIGN_TYPE instuction between \p Reg
/// and its definition, set the new register as a destination of the definition,
/// assign SPIRVType to both registers. If SpirvTy is provided, use it as
/// SPIRVType in ASSIGN_TYPE, otherwise create it from \p Ty. Defined in
/// SPIRVIRTranslator.cpp.
extern Register insertAssignInstr(Register Reg, Type *Ty, SPIRVType *SpirvTy,
                                  SPIRVGlobalRegistry *GR,
                                  MachineIRBuilder &MIB,
                                  MachineRegisterInfo &MRI);

// TODO: Move to TableGen
enum CLMemOrder {
  memory_order_relaxed = std::memory_order::memory_order_relaxed,
  memory_order_acquire = std::memory_order::memory_order_acquire,
  memory_order_release = std::memory_order::memory_order_release,
  memory_order_acq_rel = std::memory_order::memory_order_acq_rel,
  memory_order_seq_cst = std::memory_order::memory_order_seq_cst,
};

enum CLMemScope {
  memory_scope_work_item,
  memory_scope_work_group,
  memory_scope_device,
  memory_scope_all_svm_devices,
  memory_scope_sub_group
};

static MemorySemantics::MemorySemantics
getSPIRVMemSemantics(CLMemOrder clMemOrder) {
  switch (clMemOrder) {
  case memory_order_relaxed:
    return MemorySemantics::None;
  case memory_order_acquire:
    return MemorySemantics::Acquire;
  case memory_order_release:
    return MemorySemantics::Release;
  case memory_order_acq_rel:
    return MemorySemantics::AcquireRelease;
  case memory_order_seq_cst:
    return MemorySemantics::SequentiallyConsistent;
  }
  llvm_unreachable("Unknown CL memory scope");
}

static Scope::Scope getSPIRVScope(CLMemScope clScope) {
  switch (clScope) {
  case memory_scope_work_item:
    return Scope::Invocation;
  case memory_scope_work_group:
    return Scope::Workgroup;
  case memory_scope_device:
    return Scope::Device;
  case memory_scope_all_svm_devices:
    return Scope::CrossDevice;
  case memory_scope_sub_group:
    return Scope::Subgroup;
  }
  llvm_unreachable("Unknown CL memory scope");
}

/// Helper function for building an atomic load instruction.
static bool buildAtomicLoadInst(const IncomingCall *Call,
                                MachineIRBuilder &MIRBuilder,
                                SPIRVGlobalRegistry *GR) {
  SPIRVType *Int32Type = GR->getOrCreateSPIRVIntegerType(32, MIRBuilder);
  Register PtrRegister = Call->Arguments[0];
  Register ScopeRegister;

  if (Call->Arguments.size() > 1) {
    // TODO: Insert call to __translate_ocl_memory_sccope before OpAtomicLoad
    // and the function implementation. We can use Translator's output for
    // transcoding/atomic_explicit_arguments.cl as an example.
    ScopeRegister = Call->Arguments[1];
  } else {
    Scope::Scope Scope = Scope::Device;
    ScopeRegister = GR->buildConstantInt(Scope, MIRBuilder, Int32Type);
  }

  Register MemSemanticsReg;
  if (Call->Arguments.size() > 2) {
    // TODO: Insert call to __translate_ocl_memory_order before OpAtomicLoad.
    MemSemanticsReg = Call->Arguments[2];
  } else {
    int Semantics =
        MemorySemantics::SequentiallyConsistent |
        getMemSemanticsForStorageClass(GR->getPointerStorageClass(PtrRegister));
    MemSemanticsReg = GR->buildConstantInt(Semantics, MIRBuilder, Int32Type);
  }

  auto MIB = MIRBuilder.buildInstr(OpAtomicLoad)
                 .addDef(Call->ReturnRegister)
                 .addUse(GR->getSPIRVTypeID(Call->ReturnType))
                 .addUse(PtrRegister)
                 .addUse(ScopeRegister)
                 .addUse(MemSemanticsReg);

  return constrainRegOperands(MIB);
}

/// Helper function for building an atomic store instruction.
static bool buildAtomicStoreInst(const IncomingCall *Call,
                                MachineIRBuilder &MIRBuilder,
                                SPIRVGlobalRegistry *GR) {
  SPIRVType *Int32Type = GR->getOrCreateSPIRVIntegerType(32, MIRBuilder);
  Register ScopeRegister;
  Scope::Scope Scope = Scope::Device;
  if (!ScopeRegister.isValid())
    ScopeRegister = GR->buildConstantInt(Scope, MIRBuilder, Int32Type);

  Register PtrRegister = Call->Arguments[0];
  Register MemSemanticsReg;

  int Semantics =
      MemorySemantics::SequentiallyConsistent |
      getMemSemanticsForStorageClass(GR->getPointerStorageClass(PtrRegister));

  if (!MemSemanticsReg.isValid())
    MemSemanticsReg = GR->buildConstantInt(Semantics, MIRBuilder, Int32Type);

  auto MIB = MIRBuilder.buildInstr(OpAtomicStore)
                 .addUse(PtrRegister)
                 .addUse(ScopeRegister)
                 .addUse(MemSemanticsReg)
                 .addUse(Call->Arguments[1]);
  return constrainRegOperands(MIB);
}

/// Helper function for building an atomic compare-exchange instruction.
static bool buildAtomicCompareExchangeInst(const IncomingCall *Call,
                                           MachineIRBuilder &MIRBuilder,
                                           SPIRVGlobalRegistry *GR) {
  bool isCmpxchg = Call->Builtin->Name.startswith("cmpxchg");
  const auto MRI = MIRBuilder.getMRI();

  Register objectPtr = Call->Arguments[0];   // Pointer (volatile A *object)
  Register expectedArg = Call->Arguments[1]; // Comparator (C* expected)
  Register desired = Call->Arguments[2];     // Value (C desired)
  SPIRVType *spvDesiredTy = GR->getSPIRVTypeForVReg(desired);
  LLT desiredLLT = MRI->getType(desired);

  assert(GR->getSPIRVTypeForVReg(objectPtr)->getOpcode() == OpTypePointer);
  auto expectedType = GR->getSPIRVTypeForVReg(expectedArg)->getOpcode();
  assert(isCmpxchg ? expectedType == OpTypeInt : expectedType == OpTypePointer);
  assert(GR->isScalarOfType(desired, OpTypeInt));

  SPIRVType *spvObjectPtrTy = GR->getSPIRVTypeForVReg(objectPtr);
  assert(spvObjectPtrTy->getOperand(2).isReg() && "SPIRV type is expected");
  SPIRVType *spvObjectTy =
      MRI->getVRegDef(spvObjectPtrTy->getOperand(2).getReg());
  auto storageClass = static_cast<StorageClass::StorageClass>(
      spvObjectPtrTy->getOperand(1).getImm());
  auto memSemStorage = getMemSemanticsForStorageClass(storageClass);

  Register memSemEqualReg;
  Register memSemUnequalReg;
  auto memSemEqual =
      isCmpxchg ? MemorySemantics::None
                : MemorySemantics::SequentiallyConsistent | memSemStorage;
  auto memSemUnequal =
      isCmpxchg ? MemorySemantics::None
                : MemorySemantics::SequentiallyConsistent | memSemStorage;
  if (Call->Arguments.size() >= 4) {
    assert(Call->Arguments.size() >= 5 && "Need 5+ args for explicit atomic cmpxchg");
    auto memOrdEq = static_cast<CLMemOrder>(getIConstVal(Call->Arguments[3], MRI));
    auto memOrdNeq = static_cast<CLMemOrder>(getIConstVal(Call->Arguments[4], MRI));
    memSemEqual = getSPIRVMemSemantics(memOrdEq) | memSemStorage;
    memSemUnequal = getSPIRVMemSemantics(memOrdNeq) | memSemStorage;
    if (memOrdEq == memSemEqual)
      memSemEqualReg = Call->Arguments[3];
    if (memOrdNeq == memSemEqual)
      memSemUnequalReg = Call->Arguments[4];
  }
  auto I32Ty = GR->getOrCreateSPIRVIntegerType(32, MIRBuilder);
  if (!memSemEqualReg.isValid())
    memSemEqualReg = GR->buildConstantInt(memSemEqual, MIRBuilder, I32Ty);
  if (!memSemUnequalReg.isValid())
    memSemUnequalReg = GR->buildConstantInt(memSemUnequal, MIRBuilder, I32Ty);

  Register scopeReg;
  auto scope = isCmpxchg ? Scope::Workgroup : Scope::Device;
  if (Call->Arguments.size() >= 6) {
    assert(Call->Arguments.size() == 6 && "Extra args for explicit atomic cmpxchg");
    auto clScope = static_cast<CLMemScope>(getIConstVal(Call->Arguments[5], MRI));
    scope = getSPIRVScope(clScope);
    if (clScope == static_cast<unsigned>(scope))
      scopeReg = Call->Arguments[5];
  }
  if (!scopeReg.isValid())
    scopeReg = GR->buildConstantInt(scope, MIRBuilder, I32Ty);

  Register expected = isCmpxchg ? expectedArg
                                : buildLoadInst(spvDesiredTy, expectedArg,
                                            MIRBuilder, GR, LLT::scalar(32));
  MRI->setType(expected, desiredLLT);
  Register tmp =
      !isCmpxchg ? MRI->createGenericVirtualRegister(desiredLLT) : Call->ReturnRegister;
  GR->assignSPIRVTypeToVReg(spvDesiredTy, tmp, MIRBuilder.getMF());
  auto MIB = MIRBuilder.buildInstr(OpAtomicCompareExchange)
                 .addDef(tmp)
                 .addUse(GR->getSPIRVTypeID(spvObjectTy))
                 .addUse(objectPtr)
                 .addUse(scopeReg)
                 .addUse(memSemEqualReg)
                 .addUse(memSemUnequalReg)
                 .addUse(desired)
                 .addUse(expected);
  if (!isCmpxchg) {
    MIRBuilder.buildInstr(OpStore).addUse(expectedArg).addUse(tmp);
    MIRBuilder.buildICmp(CmpInst::ICMP_EQ, Call->ReturnRegister, tmp, expected);
  }
  return constrainRegOperands(MIB);
}

/// Helper function for building an atomic load instruction.
static bool buildAtomicRMWInst(const IncomingCall *Call, unsigned Opcode,
                               MachineIRBuilder &MIRBuilder,
                               SPIRVGlobalRegistry *GR) {
  const auto MRI = MIRBuilder.getMRI();
  auto Int32Type = GR->getOrCreateSPIRVIntegerType(32, MIRBuilder);

  Register ScopeRegister;
  Scope::Scope Scope = Scope::Workgroup;
  if (Call->Arguments.size() >= 4) {
    assert(Call->Arguments.size() == 4 && "Extra args for explicit atomic RMW");
    CLMemScope CLScope = static_cast<CLMemScope>(getIConstVal(Call->Arguments[5], MRI));
    Scope = getSPIRVScope(CLScope);
    if (CLScope == static_cast<unsigned>(Scope))
      ScopeRegister = Call->Arguments[5];
  }
  if (!ScopeRegister.isValid())
    ScopeRegister = GR->buildConstantInt(Scope, MIRBuilder, Int32Type);

  Register PtrRegister = Call->Arguments[0];

  Register MemSemanticsReg;
  unsigned Semantics = MemorySemantics::None;
  if (Call->Arguments.size() >= 3) {
    CLMemOrder Order = static_cast<CLMemOrder>(getIConstVal(Call->Arguments[2], MRI));
    Semantics =
        getSPIRVMemSemantics(Order) |
        getMemSemanticsForStorageClass(GR->getPointerStorageClass(PtrRegister));
    if (Order == Semantics)
      MemSemanticsReg = Call->Arguments[3];
  }
  if (!MemSemanticsReg.isValid())
    MemSemanticsReg = GR->buildConstantInt(Semantics, MIRBuilder, Int32Type);

  auto MIB = MIRBuilder.buildInstr(Opcode)
                 .addDef(Call->ReturnRegister)
                 .addUse(GR->getSPIRVTypeID(Call->ReturnType))
                 .addUse(PtrRegister)
                 .addUse(ScopeRegister)
                 .addUse(MemSemanticsReg)
                 .addUse(Call->Arguments[1]);
  return constrainRegOperands(MIB);
}

//===----------------------------------------------------------------------===//
// Implementation functions for each builtin group
//===----------------------------------------------------------------------===//

static bool generateExtInst(const IncomingCall *Call,
                            MachineIRBuilder &MIRBuilder,
                            SPIRVGlobalRegistry *GR) {
  // Lookup the extended instruction number in the TableGen records.
  const DemangledBuiltin *Builtin = Call->Builtin;
  uint32_t Number =
      lookupExtendedBuiltin(Builtin->Name, Builtin->Set)->Number;

  // Build extended instruction.
  auto MIB = MIRBuilder.buildInstr(SPIRV::OpExtInst)
                 .addDef(Call->ReturnRegister)
                 .addUse(GR->getSPIRVTypeID(Call->ReturnType))
                 .addImm(Number);

  for (auto Argument : Call->Arguments)
    MIB.addUse(Argument);

  return constrainRegOperands(MIB);
}

static bool generateRelationalInst(const IncomingCall *Call,
                            MachineIRBuilder &MIRBuilder,
                            SPIRVGlobalRegistry *GR) {
  // Lookup the instruction opcode in the TableGen records.
  const DemangledBuiltin *Builtin = Call->Builtin;
  unsigned Opcode = lookupNativeBuiltin(Builtin->Name, Builtin->Set)->Opcode;

  Register CompareRegister;
  SPIRVType *RelationType;
  std::tie(CompareRegister, RelationType) =
      buildBoolRegister(MIRBuilder, Call->ReturnType, GR);

  // Build relational instruction.
  auto MIB = MIRBuilder.buildInstr(Opcode)
                 .addDef(CompareRegister)
                 .addUse(GR->getSPIRVTypeID(RelationType));

  for (auto Argument : Call->Arguments) {
    MIB.addUse(Argument);
  }

  if (!constrainRegOperands(MIB))
    return false;

  // Build select instruction.
  return buildSelectInst(MIRBuilder, Call->ReturnRegister, CompareRegister,
                         Call->ReturnType, GR);
}

static bool generateGroupInst(const IncomingCall *Call,
                              MachineIRBuilder &MIRBuilder,
                              SPIRVGlobalRegistry *GR) {
  const DemangledBuiltin *Builtin = Call->Builtin;
  const GroupBuiltin *GroupBuiltin = lookupGroupBuiltin(Builtin->Name);

  const auto MRI = MIRBuilder.getMRI();
  Register Arg0;
  if (GroupBuiltin->HasBoolArg) {
    Register ConstRegister = Call->Arguments[0];
    auto ArgInstruction = getDefInstrMaybeConstant(ConstRegister, MRI);
    // TODO: support non-constant bool values
    assert(ArgInstruction->getOpcode() == TargetOpcode::G_CONSTANT &&
           "Only constant bool value args are supported");
    if (GR->getSPIRVTypeForVReg(Call->Arguments[0])->getOpcode() !=
        OpTypeBool) {
      auto boolTy = GR->getOrCreateSPIRVBoolType(MIRBuilder);
      Arg0 = GR->buildConstantInt(getIConstVal(ConstRegister, MRI), MIRBuilder,
                                  boolTy);
    }
  }

  Register GroupResultRegister = Call->ReturnRegister;
  SPIRVType *GroupResultType = Call->ReturnType;

  // TODO: maybe we need to check whether the result type is already boolean
  // and in this case do not insert select instruction.
  const bool HasBoolReturnTy =
      GroupBuiltin->IsElect || GroupBuiltin->IsAllOrAny ||
      GroupBuiltin->IsAllEqual || GroupBuiltin->IsLogical ||
      GroupBuiltin->IsInverseBallot || GroupBuiltin->IsBallotBitExtract;

  if (HasBoolReturnTy)
    std::tie(GroupResultRegister, GroupResultType) =
        buildBoolRegister(MIRBuilder, Call->ReturnType, GR);

  const auto I32Ty = GR->getOrCreateSPIRVIntegerType(32, MIRBuilder);
  auto Scope = Builtin->Name.startswith("sub_group") ? Scope::Subgroup
                                                     : Scope::Workgroup;
  Register ScopeRegister = GR->buildConstantInt(Scope, MIRBuilder, I32Ty);

  // Build work/sub group instruction.
  auto MIB = MIRBuilder.buildInstr(GroupBuiltin->Opcode)
                 .addDef(GroupResultRegister)
                 .addUse(GR->getSPIRVTypeID(GroupResultType))
                 .addUse(ScopeRegister);

  if (!GroupBuiltin->NoGroupOperation) {
    MIB.addImm(GroupBuiltin->GroupOperation);
  }

  if (Call->Arguments.size() > 0) {
    MIB.addUse(Arg0.isValid() ? Arg0 : Call->Arguments[0]);
    for (unsigned i = 1; i < Call->Arguments.size(); i++) {
      MIB.addUse(Call->Arguments[i]);
    }
  }
  constrainRegOperands(MIB);

  // Build select instruction.
  if (HasBoolReturnTy)
    buildSelectInst(MIRBuilder, Call->ReturnRegister, GroupResultRegister,
                    Call->ReturnType, GR);
  return true;
}

// These queries ask for a single size_t result for a given dimension index, e.g
// size_t get_global_id(uintt dimindex). In SPIR-V, the builtins corresonding to
// these values are all vec3 types, so we need to extract the correct index or
// return defaultVal (0 or 1 depending on the query). We also handle extending
// or tuncating in case size_t does not match the expected result type's
// bitwidth.
//
// For a constant index >= 3 we generate:
//  %res = OpConstant %SizeT 0
//
// For other indices we generate:
//  %g = OpVariable %ptr_V3_SizeT Input
//  OpDecorate %g BuiltIn XXX
//  OpDecorate %g LinkageAttributes "__spirv_BuiltInXXX"
//  OpDecorate %g Constant
//  %loadedVec = OpLoad %V3_SizeT %g
//
//  Then, if the index is constant < 3, we generate:
//    %res = OpCompositeExtract %SizeT %loadedVec idx
//  If the index is dynamic, we generate:
//    %tmp = OpVectorExtractDynamic %SizeT %loadedVec %idx
//    %cmp = OpULessThan %bool %idx %const_3
//    %res = OpSelect %SizeT %cmp %tmp %const_0
//
//  If the bitwidth of %res does not match the expected return type, we add an
//  extend or truncate.
//
static bool genWorkgroupQuery(const IncomingCall *Call,
                              MachineIRBuilder &MIRBuilder,
                              SPIRVGlobalRegistry *GR,
                              ::BuiltIn::BuiltIn BuiltinValue,
                              uint64_t DefaultValue) {
  Register IndexRegister = Call->Arguments[0];

  const unsigned ResultWidth = Call->ReturnType->getOperand(1).getImm();
  const unsigned int PointerSize = GR->getPointerSize();

  const auto PointerSizeType =
      GR->getOrCreateSPIRVIntegerType(PointerSize, MIRBuilder);

  const auto MRI = MIRBuilder.getMRI();
  auto IndexInstruction = getDefInstrMaybeConstant(IndexRegister, MRI);

  // Set up the final register to do truncation or extension on at the end.
  Register ToTruncate = Call->ReturnRegister;

  // If the index is constant, we can statically determine if it is in range
  bool IsConstantIndex =
      IndexInstruction->getOpcode() == TargetOpcode::G_CONSTANT;

  // If it's out of range (max dimension is 3), we can just return the constant
  // default value (0 or 1 depending on which query function)
  if (IsConstantIndex && getIConstVal(IndexRegister, MRI) >= 3) {
    Register defaultReg = Call->ReturnRegister;
    if (PointerSize != ResultWidth) {
      defaultReg = MRI->createGenericVirtualRegister(LLT::scalar(PointerSize));
      GR->assignSPIRVTypeToVReg(PointerSizeType, defaultReg,
                                MIRBuilder.getMF());
      ToTruncate = defaultReg;
    }
    auto NewRegister =
        GR->buildConstantInt(DefaultValue, MIRBuilder, PointerSizeType);
    MIRBuilder.buildCopy(defaultReg, NewRegister);
  } else { // If it could be in range, we need to load from the given builtin
    auto Vec3Ty =
        GR->getOrCreateSPIRVVectorType(PointerSizeType, 3, MIRBuilder);
    Register LoadedVector =
        buildBuiltinVariableLoad(MIRBuilder, Vec3Ty, GR, BuiltinValue,
                                 LLT::fixed_vector(3, PointerSize));

    // Set up the vreg to extract the result to (possibly a new temporary one)
    Register Extracted = Call->ReturnRegister;
    if (!IsConstantIndex || PointerSize != ResultWidth) {
      Extracted = MRI->createGenericVirtualRegister(LLT::scalar(PointerSize));
      GR->assignSPIRVTypeToVReg(PointerSizeType, Extracted, MIRBuilder.getMF());
    }

    // Use Intrinsic::spv_extractelt so dynamic vs static extraction is
    // handled later: extr = spv_extractelt LoadedVector, IndexRegister
    MachineInstrBuilder ExtractInst = MIRBuilder.buildIntrinsic(
        Intrinsic::spv_extractelt, ArrayRef<Register>{Extracted}, true);
    ExtractInst.addUse(LoadedVector).addUse(IndexRegister);

    // If the index is dynamic, need check if it's < 3, and then use a select
    if (!IsConstantIndex) {
      insertAssignInstr(Extracted, nullptr, PointerSizeType, GR, MIRBuilder,
                        *MRI);

      auto IndexType = GR->getSPIRVTypeForVReg(IndexRegister);
      auto BoolType = GR->getOrCreateSPIRVBoolType(MIRBuilder);

      Register CompareRegister =
          MRI->createGenericVirtualRegister(LLT::scalar(1));
      GR->assignSPIRVTypeToVReg(BoolType, CompareRegister, MIRBuilder.getMF());

      // Use G_ICMP to check if idxVReg < 3
      MIRBuilder.buildICmp(CmpInst::ICMP_ULT, CompareRegister, IndexRegister,
                           GR->buildConstantInt(3, MIRBuilder, IndexType));

      // Get constant for the default value (0 or 1 depending on which function)
      Register DefaultRegister =
          GR->buildConstantInt(DefaultValue, MIRBuilder, PointerSizeType);

      // Get a register for the selection result (possibly a new temporary one)
      Register SelectionResult = Call->ReturnRegister;
      if (PointerSize != ResultWidth) {
        SelectionResult =
            MRI->createGenericVirtualRegister(LLT::scalar(PointerSize));
        GR->assignSPIRVTypeToVReg(PointerSizeType, SelectionResult,
                                  MIRBuilder.getMF());
      }

      // Create the final G_SELECT to return the extracted value or the default
      MIRBuilder.buildSelect(SelectionResult, CompareRegister, Extracted,
                             DefaultRegister);
      ToTruncate = SelectionResult;
    } else {
      ToTruncate = Extracted;
    }
  }

  // Alter the result's bitwidth if it does not match the SizeT value extracted
  if (PointerSize != ResultWidth) {
    MIRBuilder.buildZExtOrTrunc(Call->ReturnRegister, ToTruncate);
  }

  return true;
}

static bool generateBuiltinVar(const IncomingCall *Call,
                               MachineIRBuilder &MIRBuilder,
                               SPIRVGlobalRegistry *GR) {
  // Lookup the builtin variable record
  const DemangledBuiltin *Builtin = Call->Builtin;
  ::BuiltIn::BuiltIn Value =
      lookupBuiltinVariable(Builtin->Name, Builtin->Set)->Value;

  if (Value == ::BuiltIn::GlobalInvocationId)
    return genWorkgroupQuery(Call, MIRBuilder, GR, Value, 0);

  // Build a load instruction for the builtin variable
  unsigned BitWidth = GR->getScalarOrVectorBitWidth(Call->ReturnType);
  LLT LLType;

  if (Call->ReturnType->getOpcode() == SPIRV::OpTypeVector)
    LLType =
        LLT::fixed_vector(Call->ReturnType->getOperand(2).getImm(), BitWidth);
  else
    LLType = LLT::scalar(BitWidth);

  return buildBuiltinVariableLoad(MIRBuilder, Call->ReturnType, GR, Value,
                                  LLType, Call->ReturnRegister);
}

static bool generateAtomicInst(const IncomingCall *Call,
                                    MachineIRBuilder &MIRBuilder,
                                    SPIRVGlobalRegistry *GR) {
  // Lookup the instruction opcode in the TableGen records.
  const DemangledBuiltin *Builtin = Call->Builtin;
  unsigned Opcode = lookupNativeBuiltin(Builtin->Name, Builtin->Set)->Opcode;

  switch (Opcode) {
  case OpAtomicLoad:
    return buildAtomicLoadInst(Call, MIRBuilder, GR);
  case OpAtomicStore:
    return buildAtomicStoreInst(Call, MIRBuilder, GR);
  case OpAtomicCompareExchange:
    return buildAtomicCompareExchangeInst(Call, MIRBuilder, GR);
  case OpAtomicIAdd:
  case OpAtomicISub:
  case OpAtomicOr:
  case OpAtomicXor:
  case OpAtomicAnd:
    return buildAtomicRMWInst(Call, Opcode, MIRBuilder, GR);
  default:
    return false;
  }
    
}

// static bool generateConvInst(const IncomingCall *Call,
//                               MachineIRBuilder &MIRBuilder,
//                               SPIRVGlobalRegistry *GR) {

// }

/// Lowers a builtin funtion call using the provided \p DemangledCall skeleton
/// and external instruction \p Set.
bool llvm::lowerBuiltin(const StringRef DemangledCall,
                        ExternalInstructionSet::InstructionSet Set,
                        MachineIRBuilder &MIRBuilder, const Register OrigRet,
                        const Type *OrigRetTy,
                        const SmallVectorImpl<Register> &Args,
                        SPIRVGlobalRegistry *GR) {

  LLVM_DEBUG(dbgs() << "Lowering builtin call: " << DemangledCall << "\n");

  // SPIR-V type for the return register
  SPIRVType *ReturnType = nullptr;
  if (OrigRetTy && !OrigRetTy->isVoidTy())
    ReturnType = GR->assignTypeToVReg(OrigRetTy, OrigRet, MIRBuilder);

  // Lookup the builtin in the TableGen records
  std::unique_ptr<const IncomingCall> Call =
      lookupBuiltin(DemangledCall, Set, OrigRet, ReturnType, Args);

  if (!Call) {
    LLVM_DEBUG(dbgs() << "Builtin record was not found!");
    return false;
  }

  verifyBuiltinArgs(Call->Builtin, Args);

  // Match the builtin with implementation based on the grouping.
  switch (Call->Builtin->Group) {
  case Extended:
    return generateExtInst(Call.get(), MIRBuilder, GR);
  case Relational:
    return generateRelationalInst(Call.get(), MIRBuilder, GR);
  case Group:
    return generateGroupInst(Call.get(), MIRBuilder, GR);
  case Variable:
    return generateBuiltinVar(Call.get(), MIRBuilder, GR);
  case Atomic:
    return generateAtomicInst(Call.get(), MIRBuilder, GR);
//  case Convert:
//    return generateConvInst(Call.get(), MIRBuilder, GR);
  default:
    break;
  }

  return false;
}