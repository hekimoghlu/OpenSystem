/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 26, 2022.
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

//===- HexagonMCInstLower.cpp - Convert Hexagon MachineInstr to an MCInst -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains code to lower Hexagon MachineInstrs to their corresponding
// MCInst records.
//
//===----------------------------------------------------------------------===//

#include "Hexagon.h"
#include "HexagonAsmPrinter.h"
#include "HexagonMachineFunctionInfo.h"
#include "llvm/Constants.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Target/Mangler.h"

using namespace llvm;

static MCOperand GetSymbolRef(const MachineOperand& MO, const MCSymbol* Symbol,
                              HexagonAsmPrinter& Printer) {
  MCContext &MC = Printer.OutContext;
  const MCExpr *ME;

  ME = MCSymbolRefExpr::Create(Symbol, MCSymbolRefExpr::VK_None, MC);

  if (!MO.isJTI() && MO.getOffset())
    ME = MCBinaryExpr::CreateAdd(ME, MCConstantExpr::Create(MO.getOffset(), MC),
                                 MC);

  return (MCOperand::CreateExpr(ME));
}

// Create an MCInst from a MachineInstr
void llvm::HexagonLowerToMC(const MachineInstr* MI, MCInst& MCI,
                            HexagonAsmPrinter& AP) {
  MCI.setOpcode(MI->getOpcode());

  for (unsigned i = 0, e = MI->getNumOperands(); i < e; i++) {
    const MachineOperand &MO = MI->getOperand(i);
    MCOperand MCO;

    switch (MO.getType()) {
    default:
      MI->dump();
      llvm_unreachable("unknown operand type");
    case MachineOperand::MO_Register:
      // Ignore all implicit register operands.
      if (MO.isImplicit()) continue;
      MCO = MCOperand::CreateReg(MO.getReg());
      break;
    case MachineOperand::MO_FPImmediate: {
      APFloat Val = MO.getFPImm()->getValueAPF();
      // FP immediates are used only when setting GPRs, so they may be dealt
      // with like regular immediates from this point on.
      MCO = MCOperand::CreateImm(*Val.bitcastToAPInt().getRawData());
      break;
    }
    case MachineOperand::MO_Immediate:
      MCO = MCOperand::CreateImm(MO.getImm());
      break;
    case MachineOperand::MO_MachineBasicBlock:
      MCO = MCOperand::CreateExpr
              (MCSymbolRefExpr::Create(MO.getMBB()->getSymbol(),
               AP.OutContext));
      break;
    case MachineOperand::MO_GlobalAddress:
      MCO = GetSymbolRef(MO, AP.Mang->getSymbol(MO.getGlobal()), AP);
      break;
    case MachineOperand::MO_ExternalSymbol:
      MCO = GetSymbolRef(MO, AP.GetExternalSymbolSymbol(MO.getSymbolName()),
                         AP);
      break;
    case MachineOperand::MO_JumpTableIndex:
      MCO = GetSymbolRef(MO, AP.GetJTISymbol(MO.getIndex()), AP);
      break;
    case MachineOperand::MO_ConstantPoolIndex:
      MCO = GetSymbolRef(MO, AP.GetCPISymbol(MO.getIndex()), AP);
      break;
    case MachineOperand::MO_BlockAddress:
      MCO = GetSymbolRef(MO, AP.GetBlockAddressSymbol(MO.getBlockAddress()),AP);
      break;
    }

    MCI.addOperand(MCO);
  }
}
