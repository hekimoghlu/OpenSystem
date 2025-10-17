/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 23, 2025.
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

//===- MemoryObject.cpp - Abstract memory interface -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/MemoryObject.h"
using namespace llvm;
  
MemoryObject::~MemoryObject() {
}

int MemoryObject::readBytes(uint64_t address,
                            uint64_t size,
                            uint8_t* buf,
                            uint64_t* copied) const {
  uint64_t current = address;
  uint64_t limit = getBase() + getExtent();

  if (current + size > limit)
    return -1;

  while (current - address < size) {
    if (readByte(current, &buf[(current - address)]))
      return -1;
    
    current++;
  }
  
  if (copied)
    *copied = current - address;
  
  return 0;
}
