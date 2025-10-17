/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 11, 2023.
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

//===-- ARMSelectionDAGInfo.h - ARM SelectionDAG Info -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the ARM subclass for TargetSelectionDAGInfo.
//
//===----------------------------------------------------------------------===//

#ifndef ARMSELECTIONDAGINFO_H
#define ARMSELECTIONDAGINFO_H

#include "MCTargetDesc/ARMAddressingModes.h"
#include "llvm/Target/TargetSelectionDAGInfo.h"

namespace llvm {

namespace ARM_AM {
  static inline ShiftOpc getShiftOpcForNode(unsigned Opcode) {
    switch (Opcode) {
    default:          return ARM_AM::no_shift;
    case ISD::SHL:    return ARM_AM::lsl;
    case ISD::SRL:    return ARM_AM::lsr;
    case ISD::SRA:    return ARM_AM::asr;
    case ISD::ROTR:   return ARM_AM::ror;
    //case ISD::ROTL:  // Only if imm -> turn into ROTR.
    // Can't handle RRX here, because it would require folding a flag into
    // the addressing mode.  :(  This causes us to miss certain things.
    //case ARMISD::RRX: return ARM_AM::rrx;
    }
  }
}  // end namespace ARM_AM

class ARMSelectionDAGInfo : public TargetSelectionDAGInfo {
  /// Subtarget - Keep a pointer to the ARMSubtarget around so that we can
  /// make the right decision when generating code for different targets.
  const ARMSubtarget *Subtarget;

public:
  explicit ARMSelectionDAGInfo(const TargetMachine &TM);
  ~ARMSelectionDAGInfo();

  virtual
  SDValue EmitTargetCodeForMemcpy(SelectionDAG &DAG, DebugLoc dl,
                                  SDValue Chain,
                                  SDValue Dst, SDValue Src,
                                  SDValue Size, unsigned Align,
                                  bool isVolatile, bool AlwaysInline,
                                  MachinePointerInfo DstPtrInfo,
                                  MachinePointerInfo SrcPtrInfo) const;

  // Adjust parameters for memset, see RTABI section 4.3.4
  virtual
  SDValue EmitTargetCodeForMemset(SelectionDAG &DAG, DebugLoc dl,
                                  SDValue Chain,
                                  SDValue Op1, SDValue Op2,
                                  SDValue Op3, unsigned Align,
                                  bool isVolatile,
                                  MachinePointerInfo DstPtrInfo) const;
};

}

#endif
