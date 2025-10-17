/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 11, 2023.
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

//===- llvm/Codegen/LinkAllCodegenComponents.h ------------------*- C++ -*-===//
//
//                      The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This header file pulls in all codegen related passes for tools like lli and
// llc that need this functionality.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_LINKALLCODEGENCOMPONENTS_H
#define LLVM_CODEGEN_LINKALLCODEGENCOMPONENTS_H

#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/SchedulerRegistry.h"
#include "llvm/CodeGen/GCs.h"
#include "llvm/Target/TargetMachine.h"
#include <cstdlib>

namespace {
  struct ForceCodegenLinking {
    ForceCodegenLinking() {
      // We must reference the passes in such a way that compilers will not
      // delete it all as dead code, even with whole program optimization,
      // yet is effectively a NO-OP. As the compiler isn't smart enough
      // to know that getenv() never returns -1, this will do the job.
      if (std::getenv("bar") != (char*) -1)
        return;

      (void) llvm::createFastRegisterAllocator();
      (void) llvm::createBasicRegisterAllocator();
      (void) llvm::createGreedyRegisterAllocator();
      (void) llvm::createDefaultPBQPRegisterAllocator();

      llvm::linkOcamlGC();
      llvm::linkShadowStackGC();

      (void) llvm::createBURRListDAGScheduler(NULL, llvm::CodeGenOpt::Default);
      (void) llvm::createSourceListDAGScheduler(NULL,llvm::CodeGenOpt::Default);
      (void) llvm::createHybridListDAGScheduler(NULL,llvm::CodeGenOpt::Default);
      (void) llvm::createFastDAGScheduler(NULL, llvm::CodeGenOpt::Default);
      (void) llvm::createDefaultScheduler(NULL, llvm::CodeGenOpt::Default);
      (void) llvm::createVLIWDAGScheduler(NULL, llvm::CodeGenOpt::Default);

    }
  } ForceCodegenLinking; // Force link by creating a global definition.
}

#endif
