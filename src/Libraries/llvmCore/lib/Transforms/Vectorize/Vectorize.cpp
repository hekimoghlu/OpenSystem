/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 3, 2024.
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

//===-- Vectorize.cpp -----------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements common infrastructure for libLLVMVectorizeOpts.a, which 
// implements several vectorization transformations over the LLVM intermediate
// representation, including the C bindings for that library.
//
//===----------------------------------------------------------------------===//

#include "llvm-c/Transforms/Vectorize.h"
#include "llvm-c/Initialization.h"
#include "llvm/InitializePasses.h"
#include "llvm/PassManager.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Transforms/Vectorize.h"

using namespace llvm;

/// initializeVectorizationPasses - Initialize all passes linked into the 
/// Vectorization library.
void llvm::initializeVectorization(PassRegistry &Registry) {
  initializeBBVectorizePass(Registry);
}

void LLVMInitializeVectorization(LLVMPassRegistryRef R) {
  initializeVectorization(*unwrap(R));
}

void LLVMAddBBVectorizePass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createBBVectorizePass());
}

