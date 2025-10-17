/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 29, 2022.
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

//===--- DominancePoint.h - Dominance points --------------------*- C++ -*-===//
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
//  This file defines types relating to local dominance calculations
//  during the emission of a function.
//
//  During the emission of a function, the LLVM IR is not well-formed enough
//  to do accurate dominance computations.  For example, a basic block may
//  appear to have a single predecessor, but that may be because a different
//  predecessor has not yet been added.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_IRGEN_DOMINANCEPOINT_H
#define LANGUAGE_IRGEN_DOMINANCEPOINT_H

#include <cassert>
#include <cstdint>

namespace language {
namespace irgen {
  class IRGenFunction;

/// An opaque class for storing keys for the dominance callback.  The
/// key is assumed to be something like a (uniqued) pointer, and a
/// null pointer is assumed to mean a non-dominating point.
class DominancePoint {
  uintptr_t Value;
  enum : uintptr_t {
    Universal = 0,
  };
  explicit DominancePoint(uintptr_t value) : Value(value) {}
public:
  explicit DominancePoint(void *value)
      : Value(reinterpret_cast<uintptr_t>(value)) {
    assert(isOrdinary());
  }

  /// Something about the definition is known to dominate all possible
  /// places that will use it.
  static DominancePoint universal() { return DominancePoint(Universal); }

  bool isOrdinary() const {
    return Value != Universal;
  }
  bool isUniversal() const {
    return Value == Universal;
  }

  template <class T> T* as() const {
    assert(isOrdinary());
    return reinterpret_cast<T*>(Value);
  }
  bool operator==(DominancePoint other) const { return Value == other.Value; }
};

/// A dominance resolver is a function that answers the question of
/// whether one dominance point dominates another.
///
/// It will only be asked this question with ordinary dominance points.
using DominanceResolverFunction = bool(*)(IRGenFunction &IGF,
                                          DominancePoint curPoint,
                                          DominancePoint definingPoint);

}
}

#endif
