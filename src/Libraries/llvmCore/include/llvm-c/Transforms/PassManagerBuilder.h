/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 17, 2022.
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
#ifndef LLVM_C_PASSMANAGERBUILDER
#define LLVM_C_PASSMANAGERBUILDER

#include "llvm-c/Core.h"

typedef struct LLVMOpaquePassManagerBuilder *LLVMPassManagerBuilderRef;

#ifdef __cplusplus
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
extern "C" {
#endif

/**
 * @defgroup LLVMCTransformsPassManagerBuilder Pass manager builder
 * @ingroup LLVMCTransforms
 *
 * @{
 */

/** See llvm::PassManagerBuilder. */
LLVMPassManagerBuilderRef LLVMPassManagerBuilderCreate(void);
void LLVMPassManagerBuilderDispose(LLVMPassManagerBuilderRef PMB);

/** See llvm::PassManagerBuilder::OptLevel. */
void
LLVMPassManagerBuilderSetOptLevel(LLVMPassManagerBuilderRef PMB,
                                  unsigned OptLevel);

/** See llvm::PassManagerBuilder::SizeLevel. */
void
LLVMPassManagerBuilderSetSizeLevel(LLVMPassManagerBuilderRef PMB,
                                   unsigned SizeLevel);

/** See llvm::PassManagerBuilder::DisableUnitAtATime. */
void
LLVMPassManagerBuilderSetDisableUnitAtATime(LLVMPassManagerBuilderRef PMB,
                                            LLVMBool Value);

/** See llvm::PassManagerBuilder::DisableUnrollLoops. */
void
LLVMPassManagerBuilderSetDisableUnrollLoops(LLVMPassManagerBuilderRef PMB,
                                            LLVMBool Value);

/** See llvm::PassManagerBuilder::DisableSimplifyLibCalls */
void
LLVMPassManagerBuilderSetDisableSimplifyLibCalls(LLVMPassManagerBuilderRef PMB,
                                                 LLVMBool Value);

/** See llvm::PassManagerBuilder::Inliner. */
void
LLVMPassManagerBuilderUseInlinerWithThreshold(LLVMPassManagerBuilderRef PMB,
                                              unsigned Threshold);

/** See llvm::PassManagerBuilder::populateFunctionPassManager. */
void
LLVMPassManagerBuilderPopulateFunctionPassManager(LLVMPassManagerBuilderRef PMB,
                                                  LLVMPassManagerRef PM);

/** See llvm::PassManagerBuilder::populateModulePassManager. */
void
LLVMPassManagerBuilderPopulateModulePassManager(LLVMPassManagerBuilderRef PMB,
                                                LLVMPassManagerRef PM);

/** See llvm::PassManagerBuilder::populateLTOPassManager. */
void LLVMPassManagerBuilderPopulateLTOPassManager(LLVMPassManagerBuilderRef PMB,
                                                  LLVMPassManagerRef PM,
                                                  bool Internalize,
                                                  bool RunInliner);

/**
 * @}
 */

#ifdef __cplusplus
}

namespace llvm {
  inline PassManagerBuilder *unwrap(LLVMPassManagerBuilderRef P) {
    return reinterpret_cast<PassManagerBuilder*>(P);
  }

  inline LLVMPassManagerBuilderRef wrap(PassManagerBuilder *P) {
    return reinterpret_cast<LLVMPassManagerBuilderRef>(P);
  }
}
#endif

#endif
