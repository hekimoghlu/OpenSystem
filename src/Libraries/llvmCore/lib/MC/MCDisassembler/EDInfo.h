/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 16, 2025.
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

//===-- EDInfo.h - LLVM Enhanced Disassembler -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//

#ifndef LLVM_EDINFO_H
#define LLVM_EDINFO_H

enum {
  EDIS_MAX_OPERANDS = 13,
  EDIS_MAX_SYNTAXES = 2
};

enum OperandTypes {
  kOperandTypeNone,
  kOperandTypeImmediate,
  kOperandTypeRegister,
  kOperandTypeX86Memory,
  kOperandTypeX86EffectiveAddress,
  kOperandTypeX86PCRelative,
  kOperandTypeARMBranchTarget,
  kOperandTypeARMSoReg,
  kOperandTypeARMSoImm,
  kOperandTypeARMRotImm,
  kOperandTypeARMSoImm2Part,
  kOperandTypeARMPredicate,
  kOperandTypeAddrModeImm12,
  kOperandTypeLdStSOReg,
  kOperandTypeARMAddrMode2,
  kOperandTypeARMAddrMode2Offset,
  kOperandTypeARMAddrMode3,
  kOperandTypeARMAddrMode3Offset,
  kOperandTypeARMAddrMode4,
  kOperandTypeARMAddrMode5,
  kOperandTypeARMAddrMode6,
  kOperandTypeARMAddrMode6Offset,
  kOperandTypeARMAddrMode7,
  kOperandTypeARMAddrModePC,
  kOperandTypeARMRegisterList,
  kOperandTypeARMDPRRegisterList,
  kOperandTypeARMSPRRegisterList,
  kOperandTypeARMTBAddrMode,
  kOperandTypeThumbITMask,
  kOperandTypeThumbAddrModeRegS1,
  kOperandTypeThumbAddrModeRegS2,
  kOperandTypeThumbAddrModeRegS4,
  kOperandTypeThumbAddrModeImmS1,
  kOperandTypeThumbAddrModeImmS2,
  kOperandTypeThumbAddrModeImmS4,
  kOperandTypeThumbAddrModeRR,
  kOperandTypeThumbAddrModeSP,
  kOperandTypeThumbAddrModePC,
  kOperandTypeThumb2AddrModeReg,
  kOperandTypeThumb2SoReg,
  kOperandTypeThumb2SoImm,
  kOperandTypeThumb2AddrModeImm8,
  kOperandTypeThumb2AddrModeImm8Offset,
  kOperandTypeThumb2AddrModeImm12,
  kOperandTypeThumb2AddrModeSoReg,
  kOperandTypeThumb2AddrModeImm8s4,
  kOperandTypeThumb2AddrModeImm8s4Offset
};

enum OperandFlags {
  kOperandFlagSource = 0x1,
  kOperandFlagTarget = 0x2
};

enum InstructionTypes {
  kInstructionTypeNone,
  kInstructionTypeMove,
  kInstructionTypeBranch,
  kInstructionTypePush,
  kInstructionTypePop,
  kInstructionTypeCall,
  kInstructionTypeReturn
};


#endif
