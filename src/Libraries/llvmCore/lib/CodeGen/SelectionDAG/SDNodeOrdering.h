/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 12, 2022.
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

//===-- llvm/CodeGen/SDNodeOrdering.h - SDNode Ordering ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the SDNodeOrdering class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_SDNODEORDERING_H
#define LLVM_CODEGEN_SDNODEORDERING_H

#include "llvm/ADT/DenseMap.h"

namespace llvm {

class SDNode;

/// SDNodeOrdering - Maps a unique (monotonically increasing) value to each
/// SDNode that roughly corresponds to the ordering of the original LLVM
/// instruction. This is used for turning off scheduling, because we'll forgo
/// the normal scheduling algorithms and output the instructions according to
/// this ordering.
class SDNodeOrdering {
  DenseMap<const SDNode*, unsigned> OrderMap;

  void operator=(const SDNodeOrdering&) LLVM_DELETED_FUNCTION;
  SDNodeOrdering(const SDNodeOrdering&) LLVM_DELETED_FUNCTION;
public:
  SDNodeOrdering() {}

  void add(const SDNode *Node, unsigned NewOrder) {
    unsigned &OldOrder = OrderMap[Node];
    if (OldOrder == 0 || (OldOrder > 0 && NewOrder < OldOrder))
      OldOrder = NewOrder;
  }
  void remove(const SDNode *Node) {
    DenseMap<const SDNode*, unsigned>::iterator Itr = OrderMap.find(Node);
    if (Itr != OrderMap.end())
      OrderMap.erase(Itr);
  }
  void clear() {
    OrderMap.clear();
  }
  unsigned getOrder(const SDNode *Node) {
    return OrderMap[Node];
  }
};

} // end llvm namespace

#endif
