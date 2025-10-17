/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 15, 2022.
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

//===-- X86.h - Top-level interface for X86 representation ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in the x86
// target library, as used by the LLVM JIT.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_X86_H
#define TARGET_X86_H

#include "MCTargetDesc/X86BaseInfo.h"
#include "MCTargetDesc/X86MCTargetDesc.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

class FunctionPass;
class JITCodeEmitter;
class X86TargetMachine;

/// createX86ISelDag - This pass converts a legalized DAG into a
/// X86-specific DAG, ready for instruction scheduling.
///
FunctionPass *createX86ISelDag(X86TargetMachine &TM,
                               CodeGenOpt::Level OptLevel);

/// createGlobalBaseRegPass - This pass initializes a global base
/// register for PIC on x86-32.
FunctionPass* createGlobalBaseRegPass();

/// createCleanupLocalDynamicTLSPass() - This pass combines multiple accesses
/// to local-dynamic TLS variables so that the TLS base address for the module
/// is only fetched once per execution path through the function.
FunctionPass *createCleanupLocalDynamicTLSPass();

/// createX86FloatingPointStackifierPass - This function returns a pass which
/// converts floating point register references and pseudo instructions into
/// floating point stack references and physical instructions.
///
FunctionPass *createX86FloatingPointStackifierPass();

/// createX86IssueVZeroUpperPass - This pass inserts AVX vzeroupper instructions
/// before each call to avoid transition penalty between functions encoded with
/// AVX and SSE.
FunctionPass *createX86IssueVZeroUpperPass();

/// createX86CodeEmitterPass - Return a pass that emits the collected X86 code
/// to the specified MCE object.
FunctionPass *createX86JITCodeEmitterPass(X86TargetMachine &TM,
                                          JITCodeEmitter &JCE);

/// createX86EmitCodeToMemory - Returns a pass that converts a register
/// allocated function into raw machine code in a dynamically
/// allocated chunk of memory.
///
FunctionPass *createEmitX86CodeToMemory();

/// createX86MaxStackAlignmentHeuristicPass - This function returns a pass
/// which determines whether the frame pointer register should be
/// reserved in case dynamic stack alignment is later required.
///
FunctionPass *createX86MaxStackAlignmentHeuristicPass();

} // End llvm namespace

#endif
