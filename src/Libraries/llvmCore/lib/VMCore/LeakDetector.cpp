/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 3, 2023.
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

//===-- LeakDetector.cpp - Implement LeakDetector interface ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the LeakDetector class.
//
//===----------------------------------------------------------------------===//

#include "LLVMContextImpl.h"
#include "llvm/Support/LeakDetector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/Threading.h"
#include "llvm/Value.h"
using namespace llvm;

static ManagedStatic<sys::SmartMutex<true> > ObjectsLock;
static ManagedStatic<LeakDetectorImpl<void> > Objects;

static void clearGarbage(LLVMContext &Context) {
  Objects->clear();
  Context.pImpl->LLVMObjects.clear();
}

void LeakDetector::addGarbageObjectImpl(void *Object) {
  sys::SmartScopedLock<true> Lock(*ObjectsLock);
  Objects->addGarbage(Object);
}

void LeakDetector::addGarbageObjectImpl(const Value *Object) {
  LLVMContextImpl *pImpl = Object->getContext().pImpl;
  pImpl->LLVMObjects.addGarbage(Object);
}

void LeakDetector::removeGarbageObjectImpl(void *Object) {
  sys::SmartScopedLock<true> Lock(*ObjectsLock);
  Objects->removeGarbage(Object);
}

void LeakDetector::removeGarbageObjectImpl(const Value *Object) {
  LLVMContextImpl *pImpl = Object->getContext().pImpl;
  pImpl->LLVMObjects.removeGarbage(Object);
}

void LeakDetector::checkForGarbageImpl(LLVMContext &Context, 
                                       const std::string &Message) {
  LLVMContextImpl *pImpl = Context.pImpl;
  sys::SmartScopedLock<true> Lock(*ObjectsLock);
  
  Objects->setName("GENERIC");
  pImpl->LLVMObjects.setName("LLVM");
  
  // use non-short-circuit version so that both checks are performed
  if (Objects->hasGarbage(Message) |
      pImpl->LLVMObjects.hasGarbage(Message))
    errs() << "\nThis is probably because you removed an object, but didn't "
           << "delete it.  Please check your code for memory leaks.\n";

  // Clear out results so we don't get duplicate warnings on
  // next call...
  clearGarbage(Context);
}
