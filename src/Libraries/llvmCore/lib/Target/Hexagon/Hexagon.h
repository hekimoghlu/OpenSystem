/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 26, 2025.
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

//=-- Hexagon.h - Top-level interface for Hexagon representation --*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in the LLVM
// Hexagon back-end.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_Hexagon_H
#define TARGET_Hexagon_H

#include "MCTargetDesc/HexagonMCTargetDesc.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {
  class FunctionPass;
  class TargetMachine;
  class MachineInstr;
  class MCInst;
  class HexagonAsmPrinter;
  class HexagonTargetMachine;
  class raw_ostream;

  FunctionPass *createHexagonISelDag(HexagonTargetMachine &TM);
  FunctionPass *createHexagonDelaySlotFillerPass(TargetMachine &TM);
  FunctionPass *createHexagonFPMoverPass(TargetMachine &TM);
  FunctionPass *createHexagonRemoveExtendOps(HexagonTargetMachine &TM);
  FunctionPass *createHexagonCFGOptimizer(HexagonTargetMachine &TM);

  FunctionPass *createHexagonSplitTFRCondSets(HexagonTargetMachine &TM);
  FunctionPass *createHexagonExpandPredSpillCode(HexagonTargetMachine &TM);

  FunctionPass *createHexagonHardwareLoops();
  FunctionPass *createHexagonPeephole();
  FunctionPass *createHexagonFixupHwLoops();
  FunctionPass *createHexagonPacketizer();
  FunctionPass *createHexagonNewValueJump();


/* TODO: object output.
  MCCodeEmitter *createHexagonMCCodeEmitter(const Target &,
                                            TargetMachine &TM,
                                            MCContext &Ctx);
*/
/* TODO: assembler input.
  TargetAsmBackend *createHexagonAsmBackend(const Target &,
                                                  const std::string &);
*/
  void HexagonLowerToMC(const MachineInstr *MI, MCInst &MCI,
                        HexagonAsmPrinter &AP);
} // end namespace llvm;

#define Hexagon_POINTER_SIZE 4

#define Hexagon_PointerSize (Hexagon_POINTER_SIZE)
#define Hexagon_PointerSize_Bits (Hexagon_POINTER_SIZE * 8)
#define Hexagon_WordSize Hexagon_PointerSize
#define Hexagon_WordSize_Bits Hexagon_PointerSize_Bits

// allocframe saves LR and FP on stack before allocating
// a new stack frame. This takes 8 bytes.
#define HEXAGON_LRFP_SIZE 8

// Normal instruction size (in bytes).
#define HEXAGON_INSTR_SIZE 4

// Maximum number of words and instructions in a packet.
#define HEXAGON_PACKET_SIZE 4

#endif
