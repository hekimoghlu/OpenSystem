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

//===-- MBlazeISelLowering.h - MBlaze DAG Lowering Interface ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that MBlaze uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#ifndef MBlazeISELLOWERING_H
#define MBlazeISELLOWERING_H

#include "MBlaze.h"
#include "MBlazeSubtarget.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/Target/TargetLowering.h"

namespace llvm {
  namespace MBlazeCC {
    enum CC {
      FIRST = 0,
      EQ,
      NE,
      GT,
      LT,
      GE,
      LE
    };

    inline static CC getOppositeCondition(CC cc) {
      switch (cc) {
      default: llvm_unreachable("Unknown condition code");
      case EQ: return NE;
      case NE: return EQ;
      case GT: return LE;
      case LT: return GE;
      case GE: return LT;
      case LE: return GE;
      }
    }

    inline static const char *MBlazeCCToString(CC cc) {
      switch (cc) {
      default: llvm_unreachable("Unknown condition code");
      case EQ: return "eq";
      case NE: return "ne";
      case GT: return "gt";
      case LT: return "lt";
      case GE: return "ge";
      case LE: return "le";
      }
    }
  }

  namespace MBlazeISD {
    enum NodeType {
      // Start the numbering from where ISD NodeType finishes.
      FIRST_NUMBER = ISD::BUILTIN_OP_END,

      // Jump and link (call)
      JmpLink,

      // Handle gp_rel (small data/bss sections) relocation.
      GPRel,

      // Select CC Pseudo Instruction
      Select_CC,

      // Wrap up multiple types of instructions
      Wrap,

      // Integer Compare
      ICmp,

      // Return from subroutine
      Ret,

      // Return from interrupt
      IRet
    };
  }

  //===--------------------------------------------------------------------===//
  // TargetLowering Implementation
  //===--------------------------------------------------------------------===//

  class MBlazeTargetLowering : public TargetLowering  {
  public:
    explicit MBlazeTargetLowering(MBlazeTargetMachine &TM);

    /// LowerOperation - Provide custom lowering hooks for some operations.
    virtual SDValue LowerOperation(SDValue Op, SelectionDAG &DAG) const;

    /// getTargetNodeName - This method returns the name of a target specific
    //  DAG node.
    virtual const char *getTargetNodeName(unsigned Opcode) const;

    /// getSetCCResultType - get the ISD::SETCC result ValueType
    EVT getSetCCResultType(EVT VT) const;

  private:
    // Subtarget Info
    const MBlazeSubtarget *Subtarget;


    // Lower Operand helpers
    SDValue LowerCallResult(SDValue Chain, SDValue InFlag,
                            CallingConv::ID CallConv, bool isVarArg,
                            const SmallVectorImpl<ISD::InputArg> &Ins,
                            DebugLoc dl, SelectionDAG &DAG,
                            SmallVectorImpl<SDValue> &InVals) const;

    // Lower Operand specifics
    SDValue LowerConstantPool(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerGlobalAddress(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerGlobalTLSAddress(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerJumpTable(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerSELECT_CC(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerVASTART(SDValue Op, SelectionDAG &DAG) const;

    virtual SDValue
      LowerFormalArguments(SDValue Chain,
                           CallingConv::ID CallConv, bool isVarArg,
                           const SmallVectorImpl<ISD::InputArg> &Ins,
                           DebugLoc dl, SelectionDAG &DAG,
                           SmallVectorImpl<SDValue> &InVals) const;

    virtual SDValue
      LowerCall(TargetLowering::CallLoweringInfo &CLI,
                SmallVectorImpl<SDValue> &InVals) const;

    virtual SDValue
      LowerReturn(SDValue Chain,
                  CallingConv::ID CallConv, bool isVarArg,
                  const SmallVectorImpl<ISD::OutputArg> &Outs,
                  const SmallVectorImpl<SDValue> &OutVals,
                  DebugLoc dl, SelectionDAG &DAG) const;

    virtual MachineBasicBlock*
      EmitCustomShift(MachineInstr *MI, MachineBasicBlock *MBB) const;

    virtual MachineBasicBlock*
      EmitCustomSelect(MachineInstr *MI, MachineBasicBlock *MBB) const;

    virtual MachineBasicBlock*
            EmitCustomAtomic(MachineInstr *MI, MachineBasicBlock *MBB) const;

    virtual MachineBasicBlock *
      EmitInstrWithCustomInserter(MachineInstr *MI,
                                  MachineBasicBlock *MBB) const;

    // Inline asm support
    ConstraintType getConstraintType(const std::string &Constraint) const;

    /// Examine constraint string and operand type and determine a weight value.
    /// The operand object must already have been set up with the operand type.
    ConstraintWeight getSingleConstraintMatchWeight(
      AsmOperandInfo &info, const char *constraint) const;

    std::pair<unsigned, const TargetRegisterClass*>
              getRegForInlineAsmConstraint(const std::string &Constraint,
              EVT VT) const;

    virtual bool isOffsetFoldingLegal(const GlobalAddressSDNode *GA) const;

    /// isFPImmLegal - Returns true if the target can instruction select the
    /// specified FP immediate natively. If false, the legalizer will
    /// materialize the FP immediate as a load from a constant pool.
    virtual bool isFPImmLegal(const APFloat &Imm, EVT VT) const;
  };
}

#endif // MBlazeISELLOWERING_H
