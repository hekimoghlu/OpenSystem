/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 12, 2023.
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
#ifndef LLVM_C_INITIALIZEPASSES_H
#define LLVM_C_INITIALIZEPASSES_H

#include "llvm-c/Core.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup LLVMCInitialization Initialization Routines
 * @ingroup LLVMC
 *
 * This module contains routines used to initialize the LLVM system.
 *
 * @{
 */

void LLVMInitializeCore(LLVMPassRegistryRef R);
void LLVMInitializeTransformUtils(LLVMPassRegistryRef R);
void LLVMInitializeScalarOpts(LLVMPassRegistryRef R);
void LLVMInitializeVectorization(LLVMPassRegistryRef R);
void LLVMInitializeInstCombine(LLVMPassRegistryRef R);
void LLVMInitializeIPO(LLVMPassRegistryRef R);
void LLVMInitializeInstrumentation(LLVMPassRegistryRef R);
void LLVMInitializeAnalysis(LLVMPassRegistryRef R);
void LLVMInitializeIPA(LLVMPassRegistryRef R);
void LLVMInitializeCodeGen(LLVMPassRegistryRef R);
void LLVMInitializeTarget(LLVMPassRegistryRef R);

/**
 * @}
 */

#ifdef __cplusplus
}
#endif

#endif
