/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 20, 2025.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */

//===-- TargetMachine.cpp -------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the LLVM-C part of TargetMachine.h
//
//===----------------------------------------------------------------------===//

#include "llvm-c/Core.h"
#include "llvm-c/Target.h"
#include "llvm-c/TargetMachine.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include <cassert>
#include <cstdlib>
#include <cstring>

using namespace llvm;



LLVMTargetRef LLVMGetFirstTarget() {
   const Target* target = &*TargetRegistry::begin();
   return wrap(target);
}
LLVMTargetRef LLVMGetNextTarget(LLVMTargetRef T) {
  return wrap(unwrap(T)->getNext());
}

const char * LLVMGetTargetName(LLVMTargetRef T) {
  return unwrap(T)->getName();
}

const char * LLVMGetTargetDescription(LLVMTargetRef T) {
  return unwrap(T)->getShortDescription();
}

LLVMBool LLVMTargetHasJIT(LLVMTargetRef T) {
  return unwrap(T)->hasJIT();
}

LLVMBool LLVMTargetHasTargetMachine(LLVMTargetRef T) {
  return unwrap(T)->hasTargetMachine();
}

LLVMBool LLVMTargetHasAsmBackend(LLVMTargetRef T) {
  return unwrap(T)->hasMCAsmBackend();
}

LLVMTargetMachineRef LLVMCreateTargetMachine(LLVMTargetRef T, char* Triple,
  char* CPU, char* Features, LLVMCodeGenOptLevel Level, LLVMRelocMode Reloc,
  LLVMCodeModel CodeModel) {
  Reloc::Model RM;
  switch (Reloc){
    case LLVMRelocStatic:
      RM = Reloc::Static;
      break;
    case LLVMRelocPIC:
      RM = Reloc::PIC_;
      break;
    case LLVMRelocDynamicNoPic:
      RM = Reloc::DynamicNoPIC;
      break;
    default:
      RM = Reloc::Default;
      break;
  }

  CodeModel::Model CM;
  switch (CodeModel) {
    case LLVMCodeModelJITDefault:
      CM = CodeModel::JITDefault;
      break;
    case LLVMCodeModelSmall:
      CM = CodeModel::Small;
      break;
    case LLVMCodeModelKernel:
      CM = CodeModel::Kernel;
      break;
    case LLVMCodeModelMedium:
      CM = CodeModel::Medium;
      break;
    case LLVMCodeModelLarge:
      CM = CodeModel::Large;
      break;
    default:
      CM = CodeModel::Default;
      break;
  }
  CodeGenOpt::Level OL;

  switch (Level) {
    case LLVMCodeGenLevelNone:
      OL = CodeGenOpt::None;
      break;
    case LLVMCodeGenLevelLess:
      OL = CodeGenOpt::Less;
      break;
    case LLVMCodeGenLevelAggressive:
      OL = CodeGenOpt::Aggressive;
      break;
    default:
      OL = CodeGenOpt::Default;
      break;
  }

  TargetOptions opt;
  return wrap(unwrap(T)->createTargetMachine(Triple, CPU, Features, opt, RM,
    CM, OL));
}


void LLVMDisposeTargetMachine(LLVMTargetMachineRef T) {
  delete unwrap(T);
}

LLVMTargetRef LLVMGetTargetMachineTarget(LLVMTargetMachineRef T) {
  const Target* target = &(unwrap(T)->getTarget());
  return wrap(target);
}

char* LLVMGetTargetMachineTriple(LLVMTargetMachineRef T) {
  std::string StringRep = unwrap(T)->getTargetTriple();
  return strdup(StringRep.c_str());
}

char* LLVMGetTargetMachineCPU(LLVMTargetMachineRef T) {
  std::string StringRep = unwrap(T)->getTargetCPU();
  return strdup(StringRep.c_str());
}

char* LLVMGetTargetMachineFeatureString(LLVMTargetMachineRef T) {
  std::string StringRep = unwrap(T)->getTargetFeatureString();
  return strdup(StringRep.c_str());
}

LLVMTargetDataRef LLVMGetTargetMachineData(LLVMTargetMachineRef T) {
  return wrap(unwrap(T)->getTargetData());
}

LLVMBool LLVMTargetMachineEmitToFile(LLVMTargetMachineRef T, LLVMModuleRef M,
  char* Filename, LLVMCodeGenFileType codegen, char** ErrorMessage) {
  TargetMachine* TM = unwrap(T);
  Module* Mod = unwrap(M);

  PassManager pass;

  std::string error;

  const TargetData* td = TM->getTargetData();

  if (!td) {
    error = "No TargetData in TargetMachine";
    *ErrorMessage = strdup(error.c_str());
    return true;
  }
  pass.add(new TargetData(*td));

  TargetMachine::CodeGenFileType ft;
  switch (codegen) {
    case LLVMAssemblyFile:
      ft = TargetMachine::CGFT_AssemblyFile;
      break;
    default:
      ft = TargetMachine::CGFT_ObjectFile;
      break;
  }
  raw_fd_ostream dest(Filename, error, raw_fd_ostream::F_Binary);
  formatted_raw_ostream destf(dest);
  if (!error.empty()) {
    *ErrorMessage = strdup(error.c_str());
    return true;
  }

  if (TM->addPassesToEmitFile(pass, destf, ft)) {
    error = "No TargetData in TargetMachine";
    *ErrorMessage = strdup(error.c_str());
    return true;
  }

  pass.run(*Mod);

  destf.flush();
  dest.flush();
  return false;
}
