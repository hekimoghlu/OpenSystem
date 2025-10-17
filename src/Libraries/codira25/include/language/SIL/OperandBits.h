/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 16, 2023.
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

//===--- OperandBits.h ----------------------------------------------------===//
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

#ifndef LANGUAGE_SIL_OPERANDBITS_H
#define LANGUAGE_SIL_OPERANDBITS_H

#include "language/SIL/SILBitfield.h"
#include "language/SIL/SILValue.h"

namespace language {

class OperandBitfield : public SILBitfield<OperandBitfield, Operand> {
  template <class, class>
  friend class SILBitfield;

  OperandBitfield *insertInto(SILFunction *function) {
    assert(function && "OperandBitField constructed with a null function");
    OperandBitfield *oldParent = function->newestAliveOperandBitfield;
    function->newestAliveOperandBitfield = this;
    return oldParent;
  }

public:
  OperandBitfield(SILFunction *function, int size)
      : SILBitfield(function, size, insertInto(function)) {}

  ~OperandBitfield() {
    assert(function->newestAliveOperandBitfield == this &&
           "BasicBlockBitfield destructed too early");
    function->newestAliveOperandBitfield = parent;
  }
};

/// A set of Operands.
///
/// For details see OperandBitfield.
class OperandSet {
  OperandBitfield bit;

public:
  OperandSet(SILFunction *function) : bit(function, 1) {}

  SILFunction *getFunction() const { return bit.getFunction(); }

  bool contains(Operand *node) const { return (bool)bit.get(node); }

  /// Returns true if \p node was not contained in the set before inserting.
  bool insert(Operand *node) {
    bool wasContained = contains(node);
    if (!wasContained) {
      bit.set(node, 1);
    }
    return !wasContained;
  }

  void erase(Operand *node) { bit.set(node, 0); }
};

using OperandSetWithSize = KnownSizeSet<OperandSet>;

} // namespace language

#endif
