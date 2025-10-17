/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 12, 2025.
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

//===-- MipsAsmPrinter.h - Mips LLVM Assembly Printer ----------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Mips Assembly printer class.
//
//===----------------------------------------------------------------------===//

#ifndef MIPSASMPRINTER_H
#define MIPSASMPRINTER_H

#include "MipsMachineFunction.h"
#include "MipsMCInstLower.h"
#include "MipsSubtarget.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {
class MCStreamer;
class MachineInstr;
class MachineBasicBlock;
class Module;
class raw_ostream;

class LLVM_LIBRARY_VISIBILITY MipsAsmPrinter : public AsmPrinter {

  void EmitInstrWithMacroNoAT(const MachineInstr *MI);

private:
  // tblgen'erated function.
  bool emitPseudoExpansionLowering(MCStreamer &OutStreamer,
                                   const MachineInstr *MI);

  // lowerOperand - Convert a MachineOperand into the equivalent MCOperand.
  bool lowerOperand(const MachineOperand &MO, MCOperand &MCOp);

public:

  const MipsSubtarget *Subtarget;
  const MipsFunctionInfo *MipsFI;
  MipsMCInstLower MCInstLowering;

  explicit MipsAsmPrinter(TargetMachine &TM,  MCStreamer &Streamer)
    : AsmPrinter(TM, Streamer), MCInstLowering(*this) {
    Subtarget = &TM.getSubtarget<MipsSubtarget>();
  }

  virtual const char *getPassName() const {
    return "Mips Assembly Printer";
  }

  virtual bool runOnMachineFunction(MachineFunction &MF);

  void EmitInstruction(const MachineInstr *MI);
  void printSavedRegsBitmask(raw_ostream &O);
  void printHex32(unsigned int Value, raw_ostream &O);
  void emitFrameDirective();
  const char *getCurrentABIString() const;
  virtual void EmitFunctionEntryLabel();
  virtual void EmitFunctionBodyStart();
  virtual void EmitFunctionBodyEnd();
  virtual bool isBlockOnlyReachableByFallthrough(const MachineBasicBlock*
                                                 MBB) const;
  bool PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                       unsigned AsmVariant, const char *ExtraCode,
                       raw_ostream &O);
  bool PrintAsmMemoryOperand(const MachineInstr *MI, unsigned OpNum,
                             unsigned AsmVariant, const char *ExtraCode,
                             raw_ostream &O);
  void printOperand(const MachineInstr *MI, int opNum, raw_ostream &O);
  void printUnsignedImm(const MachineInstr *MI, int opNum, raw_ostream &O);
  void printMemOperand(const MachineInstr *MI, int opNum, raw_ostream &O);
  void printMemOperandEA(const MachineInstr *MI, int opNum, raw_ostream &O);
  void printFCCOperand(const MachineInstr *MI, int opNum, raw_ostream &O,
                       const char *Modifier = 0);
  void EmitStartOfAsmFile(Module &M);
  virtual MachineLocation getDebugValueLocation(const MachineInstr *MI) const;
  void PrintDebugValueComment(const MachineInstr *MI, raw_ostream &OS);
};
}

#endif

