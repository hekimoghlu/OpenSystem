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

//===--- OperanDatastructures.h -------------------------------------------===//
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
//
// This file defines efficient data structures for working with Operands.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_SIL_OPERANDDATASTRUCTURES_H
#define LANGUAGE_SIL_OPERANDDATASTRUCTURES_H

#include "language/SIL/OperandBits.h"
#include "language/SIL/StackList.h"

namespace language {

/// An implementation of `toolchain::SetVector<Operand *,
///                                       StackList<Operand *>,
///                                       OperandSet>`.
///
/// Unfortunately it's not possible to use `toolchain::SetVector` directly because
/// the OperandSet and StackList constructors needs a `SILFunction`
/// argument.
///
/// Note: This class does not provide a `remove` method intentionally, because
/// it would have a O(n) complexity.
class OperandSetVector {
  StackList<Operand *> vector;
  OperandSet set;

public:
  using iterator = typename StackList<Operand *>::iterator;

  OperandSetVector(SILFunction *function) : vector(function), set(function) {}

  iterator begin() const { return vector.begin(); }
  iterator end() const { return vector.end(); }

  toolchain::iterator_range<iterator> getRange() const {
    return toolchain::make_range(begin(), end());
  }

  bool empty() const { return vector.empty(); }

  bool contains(Operand *instruction) const {
    return set.contains(instruction);
  }

  /// Returns true if \p instruction was not contained in the set before
  /// inserting.
  bool insert(Operand *instruction) {
    if (set.insert(instruction)) {
      vector.push_back(instruction);
      return true;
    }
    return false;
  }
};

/// A utility for processing instructions in a worklist.
///
/// It is basically a combination of an instruction vector and an instruction
/// set. It can be used for typical worklist-processing algorithms.
class OperandWorklist {
  StackList<Operand *> worklist;
  OperandSet visited;

public:
  /// Construct an empty worklist.
  OperandWorklist(SILFunction *function)
      : worklist(function), visited(function) {}

  /// Initialize the worklist with \p initialOperand.
  OperandWorklist(Operand *initialOperand)
      : OperandWorklist(initialOperand->getUser()->getFunction()) {
    push(initialOperand);
  }

  /// Pops the last added element from the worklist or returns null, if the
  /// worklist is empty.
  Operand *pop() {
    if (worklist.empty())
      return nullptr;
    return worklist.pop_back_val();
  }

  /// Pushes \p operand onto the worklist if \p operand has never been
  /// push before.
  bool pushIfNotVisited(Operand *operand) {
    if (visited.insert(operand)) {
      worklist.push_back(operand);
      return true;
    }
    return false;
  }

  /// Pushes the operands of all uses of \p instruction onto the worklist if the
  /// operands have never been pushed before. Returns \p true if we inserted
  /// /any/ values.
  ///
  /// This is a bulk convenience API.
  bool pushResultOperandsIfNotVisited(SILInstruction *inst) {
    bool insertedOperand = false;
    for (auto result : inst->getResults()) {
      for (auto *use : result->getUses()) {
        insertedOperand |= pushIfNotVisited(use);
      }
    }
    return insertedOperand;
  }

  /// Like `pushIfNotVisited`, but requires that \p operand has never been
  /// on the worklist before.
  void push(Operand *operand) {
    assert(!visited.contains(operand));
    visited.insert(operand);
    worklist.push_back(operand);
  }

  /// Like `pop`, but marks the returned operand as "unvisited". This means,
  /// that the operand can be pushed onto the worklist again.
  Operand *popAndForget() {
    if (worklist.empty())
      return nullptr;
    Operand *operand = worklist.pop_back_val();
    visited.erase(operand);
    return operand;
  }

  /// Returns true if \p operand was visited, i.e. has been added to the
  /// worklist.
  bool isVisited(Operand *operand) const { return visited.contains(operand); }
};

} // namespace language

#endif
