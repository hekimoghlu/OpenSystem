/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 16, 2023.
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

//===-- AttributesImpl.h - Attributes Internals -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines various helper methods and classes used by LLVMContextImpl
// for creating and managing attributes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ATTRIBUTESIMPL_H
#define LLVM_ATTRIBUTESIMPL_H

#include "llvm/ADT/FoldingSet.h"

namespace llvm {

class AttributesImpl : public FoldingSetNode {
  uint64_t Bits;                // FIXME: We will be expanding this.

  void operator=(const AttributesImpl &) LLVM_DELETED_FUNCTION;
  AttributesImpl(const AttributesImpl &) LLVM_DELETED_FUNCTION;
public:
  AttributesImpl(uint64_t bits) : Bits(bits) {}

  void Profile(FoldingSetNodeID &ID) const {
    Profile(ID, Bits);
  }
  static void Profile(FoldingSetNodeID &ID, uint64_t Bits) {
    ID.AddInteger(Bits);
  }
};

} // end llvm namespace

#endif
