/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 26, 2025.
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

//===-- StringPool.cpp - Interned string pool -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the StringPool class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/StringPool.h"
#include "llvm/ADT/StringRef.h"

using namespace llvm;

StringPool::StringPool() {}

StringPool::~StringPool() {
  assert(InternTable.empty() && "PooledStringPtr leaked!");
}

PooledStringPtr StringPool::intern(StringRef Key) {
  table_t::iterator I = InternTable.find(Key);
  if (I != InternTable.end())
    return PooledStringPtr(&*I);
  
  entry_t *S = entry_t::Create(Key.begin(), Key.end());
  S->getValue().Pool = this;
  InternTable.insert(S);
  
  return PooledStringPtr(S);
}
