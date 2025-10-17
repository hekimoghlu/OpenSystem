/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 29, 2023.
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

//===-- ARMAsmPrinter.h - Print machine code to an ARM .s file --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// ARM Assembly printer class.
//
//===----------------------------------------------------------------------===//

#ifndef ARMASMPRINTER_H
#define ARMASMPRINTER_H

#include "ARM.h"
#include "ARMTargetMachine.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

class MCOperand;

namespace ARM {
  enum DW_ISA {
    DW_ISA_ARM_thumb = 1,
    DW_ISA_ARM_arm = 2
  };
}

class LLVM_LIBRARY_VISIBILITY ARMAsmPrinter : public AsmPrinter {

  /// Subtarget - Keep a pointer to the ARMSubtarget around so that we can
  /// make the right decision when printing asm code for different targets.
  const ARMSubtarget *Subtarget;

  /// AFI - Keep a pointer to ARMFunctionInfo for the current
  /// MachineFunction.
  ARMFunctionInfo *AFI;

  /// MCP - Keep a pointer to constantpool entries of the current
  /// MachineFunction.
  const MachineConstantPool *MCP;

  /// InConstantPool - Maintain state when emitting a sequence of constant
  /// pool entries so we can properly mark them as data regions.
  bool InConstantPool;
public:
  explicit ARMAsmPrinter(TargetMachine &TM, MCStreamer &Streamer)
    : AsmPrinter(TM, Streamer), AFI(NULL), MCP(NULL), InConstantPool(false) {
      Subtarget = &TM.getSubtarget<ARMSubtarget>();
    }

  virtual const char *getPassName() const {
    return "ARM Assembly Printer";
  }

  void printOperand(const MachineInstr *MI, int OpNum, raw_ostream &O,
                    const char *Modifier = 0);

  virtual bool PrintAsmOperand(const MachineInstr *MI, unsigned OpNum,
                               unsigned AsmVariant, const char *ExtraCode,
                               raw_ostream &O);
  virtual bool PrintAsmMemoryOperand(const MachineInstr *MI, unsigned OpNum,
                                     unsigned AsmVariant,
                                     const char *ExtraCode, raw_ostream &O);

  void EmitJumpTable(const MachineInstr *MI);
  void EmitJump2Table(const MachineInstr *MI);
  virtual void EmitInstruction(const MachineInstr *MI);
  bool runOnMachineFunction(MachineFunction &F);

  virtual void EmitConstantPool() {} // we emit constant pools customly!
  virtual void EmitFunctionBodyEnd();
  virtual void EmitFunctionEntryLabel();
  void EmitStartOfAsmFile(Module &M);
  void EmitEndOfAsmFile(Module &M);
  void EmitXXStructor(const Constant *CV);

  // lowerOperand - Convert a MachineOperand into the equivalent MCOperand.
  bool lowerOperand(const MachineOperand &MO, MCOperand &MCOp);

private:
  // Helpers for EmitStartOfAsmFile() and EmitEndOfAsmFile()
  void emitAttributes();

  // Helper for ELF .o only
  void emitARMAttributeSection();

  // Generic helper used to emit e.g. ARMv5 mul pseudos
  void EmitPatchedInstruction(const MachineInstr *MI, unsigned TargetOpc);

  void EmitUnwindingInstruction(const MachineInstr *MI);

  // emitPseudoExpansionLowering - tblgen'erated.
  bool emitPseudoExpansionLowering(MCStreamer &OutStreamer,
                                   const MachineInstr *MI);

public:
  void PrintDebugValueComment(const MachineInstr *MI, raw_ostream &OS);

  MachineLocation getDebugValueLocation(const MachineInstr *MI) const;

  /// EmitDwarfRegOp - Emit dwarf register operation.
  virtual void EmitDwarfRegOp(const MachineLocation &MLoc) const;

  virtual unsigned getISAEncoding() {
    // ARM/Darwin adds ISA to the DWARF info for each function.
    if (!Subtarget->isTargetDarwin())
      return 0;
    return Subtarget->isThumb() ?
      ARM::DW_ISA_ARM_thumb : ARM::DW_ISA_ARM_arm;
  }

  MCOperand GetSymbolRef(const MachineOperand &MO, const MCSymbol *Symbol);
  MCSymbol *GetARMSetPICJumpTableLabel2(unsigned uid, unsigned uid2,
                                        const MachineBasicBlock *MBB) const;
  MCSymbol *GetARMJTIPICJumpTableLabel2(unsigned uid, unsigned uid2) const;

  MCSymbol *GetARMSJLJEHLabel(void) const;

  MCSymbol *GetARMGVSymbol(const GlobalValue *GV);

  /// EmitMachineConstantPoolValue - Print a machine constantpool value to
  /// the .s file.
  virtual void EmitMachineConstantPoolValue(MachineConstantPoolValue *MCPV);
};
} // end namespace llvm

#endif
