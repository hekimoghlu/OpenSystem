/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 3, 2025.
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

//===- llvm/Support/PredIteratorCache.h - pred_iterator Cache ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the PredIteratorCache class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Allocator.h"
#include "llvm/Support/CFG.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#ifndef LLVM_SUPPORT_PREDITERATORCACHE_H
#define LLVM_SUPPORT_PREDITERATORCACHE_H

namespace llvm {

  /// PredIteratorCache - This class is an extremely trivial cache for
  /// predecessor iterator queries.  This is useful for code that repeatedly
  /// wants the predecessor list for the same blocks.
  class PredIteratorCache {
    /// BlockToPredsMap - Pointer to null-terminated list.
    DenseMap<BasicBlock*, BasicBlock**> BlockToPredsMap;
    DenseMap<BasicBlock*, unsigned> BlockToPredCountMap;

    /// Memory - This is the space that holds cached preds.
    BumpPtrAllocator Memory;
  public:

    /// GetPreds - Get a cached list for the null-terminated predecessor list of
    /// the specified block.  This can be used in a loop like this:
    ///   for (BasicBlock **PI = PredCache->GetPreds(BB); *PI; ++PI)
    ///      use(*PI);
    /// instead of:
    /// for (pred_iterator PI = pred_begin(BB), E = pred_end(BB); PI != E; ++PI)
    BasicBlock **GetPreds(BasicBlock *BB) {
      BasicBlock **&Entry = BlockToPredsMap[BB];
      if (Entry) return Entry;

      SmallVector<BasicBlock*, 32> PredCache(pred_begin(BB), pred_end(BB));
      PredCache.push_back(0); // null terminator.
      
      BlockToPredCountMap[BB] = PredCache.size()-1;

      Entry = Memory.Allocate<BasicBlock*>(PredCache.size());
      std::copy(PredCache.begin(), PredCache.end(), Entry);
      return Entry;
    }
    
    unsigned GetNumPreds(BasicBlock *BB) {
      GetPreds(BB);
      return BlockToPredCountMap[BB];
    }

    /// clear - Remove all information.
    void clear() {
      BlockToPredsMap.clear();
      BlockToPredCountMap.clear();
      Memory.Reset();
    }
  };
} // end namespace llvm

#endif
