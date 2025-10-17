/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 28, 2025.
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
#ifndef LLVM_C_TRANSFORMS_SCALAR_H
#define LLVM_C_TRANSFORMS_SCALAR_H

#include "llvm-c/Core.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup LLVMCTransformsScalar Scalar transformations
 * @ingroup LLVMCTransforms
 *
 * @{
 */

/** See llvm::createAggressiveDCEPass function. */
void LLVMAddAggressiveDCEPass(LLVMPassManagerRef PM);

/** See llvm::createCFGSimplificationPass function. */
void LLVMAddCFGSimplificationPass(LLVMPassManagerRef PM);

/** See llvm::createDeadStoreEliminationPass function. */
void LLVMAddDeadStoreEliminationPass(LLVMPassManagerRef PM);

/** See llvm::createGVNPass function. */
void LLVMAddGVNPass(LLVMPassManagerRef PM);

/** See llvm::createIndVarSimplifyPass function. */
void LLVMAddIndVarSimplifyPass(LLVMPassManagerRef PM);

/** See llvm::createInstructionCombiningPass function. */
void LLVMAddInstructionCombiningPass(LLVMPassManagerRef PM);

/** See llvm::createJumpThreadingPass function. */
void LLVMAddJumpThreadingPass(LLVMPassManagerRef PM);

/** See llvm::createLICMPass function. */
void LLVMAddLICMPass(LLVMPassManagerRef PM);

/** See llvm::createLoopDeletionPass function. */
void LLVMAddLoopDeletionPass(LLVMPassManagerRef PM);

/** See llvm::createLoopIdiomPass function */
void LLVMAddLoopIdiomPass(LLVMPassManagerRef PM);

/** See llvm::createLoopRotatePass function. */
void LLVMAddLoopRotatePass(LLVMPassManagerRef PM);

/** See llvm::createLoopUnrollPass function. */
void LLVMAddLoopUnrollPass(LLVMPassManagerRef PM);

/** See llvm::createLoopUnswitchPass function. */
void LLVMAddLoopUnswitchPass(LLVMPassManagerRef PM);

/** See llvm::createMemCpyOptPass function. */
void LLVMAddMemCpyOptPass(LLVMPassManagerRef PM);

/** See llvm::createPromoteMemoryToRegisterPass function. */
void LLVMAddPromoteMemoryToRegisterPass(LLVMPassManagerRef PM);

/** See llvm::createReassociatePass function. */
void LLVMAddReassociatePass(LLVMPassManagerRef PM);

/** See llvm::createSCCPPass function. */
void LLVMAddSCCPPass(LLVMPassManagerRef PM);

/** See llvm::createScalarReplAggregatesPass function. */
void LLVMAddScalarReplAggregatesPass(LLVMPassManagerRef PM);

/** See llvm::createScalarReplAggregatesPass function. */
void LLVMAddScalarReplAggregatesPassSSA(LLVMPassManagerRef PM);

/** See llvm::createScalarReplAggregatesPass function. */
void LLVMAddScalarReplAggregatesPassWithThreshold(LLVMPassManagerRef PM,
                                                  int Threshold);

/** See llvm::createSimplifyLibCallsPass function. */
void LLVMAddSimplifyLibCallsPass(LLVMPassManagerRef PM);

/** See llvm::createTailCallEliminationPass function. */
void LLVMAddTailCallEliminationPass(LLVMPassManagerRef PM);

/** See llvm::createConstantPropagationPass function. */
void LLVMAddConstantPropagationPass(LLVMPassManagerRef PM);

/** See llvm::demotePromoteMemoryToRegisterPass function. */
void LLVMAddDemoteMemoryToRegisterPass(LLVMPassManagerRef PM);

/** See llvm::createVerifierPass function. */
void LLVMAddVerifierPass(LLVMPassManagerRef PM);

/** See llvm::createCorrelatedValuePropagationPass function */
void LLVMAddCorrelatedValuePropagationPass(LLVMPassManagerRef PM);

/** See llvm::createEarlyCSEPass function */
void LLVMAddEarlyCSEPass(LLVMPassManagerRef PM);

/** See llvm::createLowerExpectIntrinsicPass function */
void LLVMAddLowerExpectIntrinsicPass(LLVMPassManagerRef PM);

/** See llvm::createTypeBasedAliasAnalysisPass function */
void LLVMAddTypeBasedAliasAnalysisPass(LLVMPassManagerRef PM);

/** See llvm::createBasicAliasAnalysisPass function */
void LLVMAddBasicAliasAnalysisPass(LLVMPassManagerRef PM);

/**
 * @}
 */

#ifdef __cplusplus
}
#endif /* defined(__cplusplus) */

#endif
