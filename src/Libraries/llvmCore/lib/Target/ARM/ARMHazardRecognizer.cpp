/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 22, 2022.
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

//===-- ARMHazardRecognizer.cpp - ARM postra hazard recognizer ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ARMHazardRecognizer.h"
#include "ARMBaseInstrInfo.h"
#include "ARMBaseRegisterInfo.h"
#include "ARMSubtarget.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/Target/TargetRegisterInfo.h"
using namespace llvm;

static bool hasRAWHazard(MachineInstr *DefMI, MachineInstr *MI,
                         const TargetRegisterInfo &TRI) {
  // FIXME: Detect integer instructions properly.
  const MCInstrDesc &MCID = MI->getDesc();
  unsigned Domain = MCID.TSFlags & ARMII::DomainMask;
  if (MI->mayStore())
    return false;
  unsigned Opcode = MCID.getOpcode();
  if (Opcode == ARM::VMOVRS || Opcode == ARM::VMOVRRD)
    return false;
  if ((Domain & ARMII::DomainVFP) || (Domain & ARMII::DomainNEON))
    return MI->readsRegister(DefMI->getOperand(0).getReg(), &TRI);
  return false;
}

ScheduleHazardRecognizer::HazardType
ARMHazardRecognizer::getHazardType(SUnit *SU, int Stalls) {
  assert(Stalls == 0 && "ARM hazards don't support scoreboard lookahead");

  MachineInstr *MI = SU->getInstr();

  if (!MI->isDebugValue()) {
    // Look for special VMLA / VMLS hazards. A VMUL / VADD / VSUB following
    // a VMLA / VMLS will cause 4 cycle stall.
    const MCInstrDesc &MCID = MI->getDesc();
    if (LastMI && (MCID.TSFlags & ARMII::DomainMask) != ARMII::DomainGeneral) {
      MachineInstr *DefMI = LastMI;
      const MCInstrDesc &LastMCID = LastMI->getDesc();
      // Skip over one non-VFP / NEON instruction.
      if (!LastMI->isBarrier() &&
          // On A9, AGU and NEON/FPU are muxed.
          !(STI.isLikeA9() && (LastMI->mayLoad() || LastMI->mayStore())) &&
          (LastMCID.TSFlags & ARMII::DomainMask) == ARMII::DomainGeneral) {
        MachineBasicBlock::iterator I = LastMI;
        if (I != LastMI->getParent()->begin()) {
          I = llvm::prior(I);
          DefMI = &*I;
        }
      }

      if (TII.isFpMLxInstruction(DefMI->getOpcode()) &&
          (TII.canCauseFpMLxStall(MI->getOpcode()) ||
           hasRAWHazard(DefMI, MI, TRI))) {
        // Try to schedule another instruction for the next 4 cycles.
        if (FpMLxStalls == 0)
          FpMLxStalls = 4;
        return Hazard;
      }
    }
  }

  return ScoreboardHazardRecognizer::getHazardType(SU, Stalls);
}

void ARMHazardRecognizer::Reset() {
  LastMI = 0;
  FpMLxStalls = 0;
  ScoreboardHazardRecognizer::Reset();
}

void ARMHazardRecognizer::EmitInstruction(SUnit *SU) {
  MachineInstr *MI = SU->getInstr();
  if (!MI->isDebugValue()) {
    LastMI = MI;
    FpMLxStalls = 0;
  }

  ScoreboardHazardRecognizer::EmitInstruction(SU);
}

void ARMHazardRecognizer::AdvanceCycle() {
  if (FpMLxStalls && --FpMLxStalls == 0)
    // Stalled for 4 cycles but still can't schedule any other instructions.
    LastMI = 0;
  ScoreboardHazardRecognizer::AdvanceCycle();
}

void ARMHazardRecognizer::RecedeCycle() {
  llvm_unreachable("reverse ARM hazard checking unsupported");
}
