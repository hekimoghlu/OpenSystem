/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 10, 2023.
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

//===-- ManagedStringPool.h - Managed String Pool ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// The strings allocated from a managed string pool are owned by the string
// pool and will be deleted together with the managed string pool.
//
//===----------------------------------------------------------------------===//


#ifndef LLVM_SUPPORT_MANAGED_STRING_H
#define LLVM_SUPPORT_MANAGED_STRING_H

#include "llvm/ADT/SmallVector.h"
#include <string>

namespace llvm {

/// ManagedStringPool - The strings allocated from a managed string pool are
/// owned by the string pool and will be deleted together with the managed
/// string pool.
class ManagedStringPool {
  SmallVector<std::string *, 8> Pool;

public:
  ManagedStringPool() {}
  ~ManagedStringPool() {
    SmallVector<std::string *, 8>::iterator Current = Pool.begin();
    while (Current != Pool.end()) {
      delete *Current;
      Current++;
    }
  }

  std::string *getManagedString(const char *S) {
    std::string *Str = new std::string(S);
    Pool.push_back(Str);
    return Str;
  }
};

}

#endif
