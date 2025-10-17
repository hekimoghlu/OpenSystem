/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 4, 2022.
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

//===--- IndexTrie - Trie for a sequence of integer indices ----*- C++ -*-===//
//
// Copyright (c) NeXTHub Corporation. All rights reserved.
// DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
//
// This code is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// version 2 for more details (a copy is included in the LICENSE file that
// accompanied this code).
//
// Author(-s): Tunjay Akbarli
//

//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_SILOPTIMIZER_UTILS_INDEXTREE_H
#define LANGUAGE_SILOPTIMIZER_UTILS_INDEXTREE_H

#include "language/Basic/Toolchain.h"
#include "toolchain/ADT/ArrayRef.h"
#include "toolchain/ADT/SmallVector.h"
#include <algorithm>

namespace language {

// Trie node representing a sequence of unsigned integer indices.
class IndexTrieNode {
public:
  static const int RootIndex = std::numeric_limits<int>::min();

private:
  int Index;
  toolchain::SmallVector<IndexTrieNode*, 8> Children;
  IndexTrieNode *Parent;

public:
  IndexTrieNode() : Index(RootIndex), Parent(nullptr) {}

  explicit IndexTrieNode(int V, IndexTrieNode *P) : Index(V), Parent(P) {}

  IndexTrieNode(IndexTrieNode &) =delete;
  IndexTrieNode &operator=(const IndexTrieNode&) =delete;

  ~IndexTrieNode() {
    for (auto *N : Children)
      delete N;
  }

  bool isRoot() const { return Index == RootIndex; }

  bool isLeaf() const { return Children.empty(); }

  int getIndex() const { return Index; }

  IndexTrieNode *getChild(int Idx) {
    assert(Idx != RootIndex);

    auto I =
        std::lower_bound(Children.begin(), Children.end(), Idx,
                         [](IndexTrieNode *a, int i) { return a->Index < i; });
    if (I != Children.end() && (*I)->Index == Idx)
      return *I;
    auto *N = new IndexTrieNode(Idx, this);
    Children.insert(I, N);
    return N;
  }

  ArrayRef<IndexTrieNode*> getChildren() const { return Children; }

  IndexTrieNode *getParent() const { return Parent; }

  /// Returns true when the sequence of indices represented by this
  /// node is a prefix of the sequence represented by the passed-in node.
  bool isPrefixOf(const IndexTrieNode *Other) const {
    const IndexTrieNode *I = Other;

    do {
      if (this == I)
        return true;

      I = I->getParent();
    } while (I);

    return false;
  }
};

} // end namespace language

#endif
