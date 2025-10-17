/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 23, 2023.
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

//===-- CodeGen.cpp -------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the common initialization routines for the
// CodeGen library.
//
//===----------------------------------------------------------------------===//

#include "llvm/InitializePasses.h"
#include "llvm-c/Initialization.h"

using namespace llvm;

/// initializeCodeGen - Initialize all passes linked into the CodeGen library.
void llvm::initializeCodeGen(PassRegistry &Registry) {
  initializeBranchFolderPassPass(Registry);
  initializeCalculateSpillWeightsPass(Registry);
  initializeCodePlacementOptPass(Registry);
  initializeDeadMachineInstructionElimPass(Registry);
  initializeEarlyIfConverterPass(Registry);
  initializeExpandPostRAPass(Registry);
  initializeExpandISelPseudosPass(Registry);
  initializeFinalizeMachineBundlesPass(Registry);
  initializeGCMachineCodeAnalysisPass(Registry);
  initializeGCModuleInfoPass(Registry);
  initializeIfConverterPass(Registry);
  initializeLiveDebugVariablesPass(Registry);
  initializeLiveIntervalsPass(Registry);
  initializeLiveStacksPass(Registry);
  initializeLiveVariablesPass(Registry);
  initializeLocalStackSlotPassPass(Registry);
  initializeMachineBlockFrequencyInfoPass(Registry);
  initializeMachineBlockPlacementPass(Registry);
  initializeMachineBlockPlacementStatsPass(Registry);
  initializeMachineCopyPropagationPass(Registry);
  initializeMachineCSEPass(Registry);
  initializeMachineDominatorTreePass(Registry);
  initializeMachinePostDominatorTreePass(Registry);
  initializeMachineLICMPass(Registry);
  initializeMachineLoopInfoPass(Registry);
  initializeMachineModuleInfoPass(Registry);
  initializeMachineSchedulerPass(Registry);
  initializeMachineSinkingPass(Registry);
  initializeMachineVerifierPassPass(Registry);
  initializeOptimizePHIsPass(Registry);
  initializePHIEliminationPass(Registry);
  initializePeepholeOptimizerPass(Registry);
  initializePostRASchedulerPass(Registry);
  initializeProcessImplicitDefsPass(Registry);
  initializePEIPass(Registry);
  initializeRegisterCoalescerPass(Registry);
  initializeSlotIndexesPass(Registry);
  initializeStackProtectorPass(Registry);
  initializeStackColoringPass(Registry);
  initializeStackSlotColoringPass(Registry);
  initializeStrongPHIEliminationPass(Registry);
  initializeTailDuplicatePassPass(Registry);
  initializeTargetPassConfigPass(Registry);
  initializeTwoAddressInstructionPassPass(Registry);
  initializeUnpackMachineBundlesPass(Registry);
  initializeUnreachableBlockElimPass(Registry);
  initializeUnreachableMachineBlockElimPass(Registry);
  initializeVirtRegMapPass(Registry);
  initializeVirtRegRewriterPass(Registry);
  initializeLowerIntrinsicsPass(Registry);
  initializeMachineFunctionPrinterPassPass(Registry);
}

void LLVMInitializeCodeGen(LLVMPassRegistryRef R) {
  initializeCodeGen(*unwrap(R));
}
