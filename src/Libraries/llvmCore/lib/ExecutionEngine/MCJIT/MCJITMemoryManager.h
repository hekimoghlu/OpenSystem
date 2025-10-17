/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 7, 2022.
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

//===-- MCJITMemoryManager.h - Definition for the Memory Manager ---C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_EXECUTIONENGINE_MCJITMEMORYMANAGER_H
#define LLVM_LIB_EXECUTIONENGINE_MCJITMEMORYMANAGER_H

#include "llvm/Module.h"
#include "llvm/ExecutionEngine/JITMemoryManager.h"
#include "llvm/ExecutionEngine/RuntimeDyld.h"
#include <assert.h>

namespace llvm {

// The MCJIT memory manager is a layer between the standard JITMemoryManager
// and the RuntimeDyld interface that maps objects, by name, onto their
// matching LLVM IR counterparts in the module(s) being compiled.
class MCJITMemoryManager : public RTDyldMemoryManager {
  virtual void anchor();
  OwningPtr<JITMemoryManager> JMM;

public:
  MCJITMemoryManager(JITMemoryManager *jmm) :
    JMM(jmm?jmm:JITMemoryManager::CreateDefaultMemManager()) {}

  uint8_t *allocateDataSection(uintptr_t Size, unsigned Alignment,
                               unsigned SectionID) {
    return JMM->allocateDataSection(Size, Alignment, SectionID);
  }

  uint8_t *allocateCodeSection(uintptr_t Size, unsigned Alignment,
                               unsigned SectionID) {
    return JMM->allocateCodeSection(Size, Alignment, SectionID);
  }

  virtual void *getPointerToNamedFunction(const std::string &Name,
                                          bool AbortOnFailure = true) {
    return JMM->getPointerToNamedFunction(Name, AbortOnFailure);
  }

};

} // End llvm namespace

#endif
