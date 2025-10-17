/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 5, 2025.
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

//===-- NVPTX.h - Top-level interface for NVPTX representation --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in
// the LLVM NVPTX back-end.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_NVPTX_H
#define LLVM_TARGET_NVPTX_H

#include "llvm/Value.h"
#include "llvm/Module.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Target/TargetMachine.h"
#include "MCTargetDesc/NVPTXBaseInfo.h"
#include <cassert>
#include <iosfwd>

namespace llvm {
class NVPTXTargetMachine;
class FunctionPass;
class formatted_raw_ostream;

namespace NVPTXCC {
enum CondCodes {
  EQ,
  NE,
  LT,
  LE,
  GT,
  GE
};
}

inline static const char *NVPTXCondCodeToString(NVPTXCC::CondCodes CC) {
  switch (CC) {
  case NVPTXCC::NE:  return "ne";
  case NVPTXCC::EQ:   return "eq";
  case NVPTXCC::LT:   return "lt";
  case NVPTXCC::LE:  return "le";
  case NVPTXCC::GT:  return "gt";
  case NVPTXCC::GE:   return "ge";
  }
  llvm_unreachable("Unknown condition code");
}

FunctionPass *createNVPTXISelDag(NVPTXTargetMachine &TM,
                                 llvm::CodeGenOpt::Level OptLevel);
FunctionPass *createVectorElementizePass(NVPTXTargetMachine &);
FunctionPass *createLowerStructArgsPass(NVPTXTargetMachine &);
FunctionPass *createNVPTXReMatPass(NVPTXTargetMachine &);
FunctionPass *createNVPTXReMatBlockPass(NVPTXTargetMachine &);

bool isImageOrSamplerVal(const Value *, const Module *);

extern Target TheNVPTXTarget32;
extern Target TheNVPTXTarget64;

namespace NVPTX
{
enum DrvInterface {
  NVCL,
  CUDA,
  TEST
};

// A field inside TSFlags needs a shift and a mask. The usage is
// always as follows :
// ((TSFlags & fieldMask) >> fieldShift)
// The enum keeps the mask, the shift, and all valid values of the
// field in one place.
enum VecInstType {
  VecInstTypeShift = 0,
  VecInstTypeMask = 0xF,

  VecNOP = 0,
  VecLoad = 1,
  VecStore = 2,
  VecBuild = 3,
  VecShuffle = 4,
  VecExtract = 5,
  VecInsert = 6,
  VecDest = 7,
  VecOther = 15
};

enum SimpleMove {
  SimpleMoveMask = 0x10,
  SimpleMoveShift = 4
};
enum LoadStore {
  isLoadMask = 0x20,
  isLoadShift = 5,
  isStoreMask = 0x40,
  isStoreShift = 6
};

namespace PTXLdStInstCode {
enum AddressSpace{
  GENERIC = 0,
  GLOBAL = 1,
  CONSTANT = 2,
  SHARED = 3,
  PARAM = 4,
  LOCAL = 5
};
enum FromType {
  Unsigned = 0,
  Signed,
  Float
};
enum VecType {
  Scalar = 1,
  V2 = 2,
  V4 = 4
};
}
}
} // end namespace llvm;

// Defines symbolic names for NVPTX registers.  This defines a mapping from
// register name to register number.
#define GET_REGINFO_ENUM
#include "NVPTXGenRegisterInfo.inc"

// Defines symbolic names for the NVPTX instructions.
#define GET_INSTRINFO_ENUM
#include "NVPTXGenInstrInfo.inc"

#endif
