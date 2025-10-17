/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 16, 2025.
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

//===--- Temporary.h - A temporary allocation -------------------*- C++ -*-===//
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
//  This file defines the Temporary and TemporarySet classes.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_IRGEN_TEMPORARY_H
#define LANGUAGE_IRGEN_TEMPORARY_H

#include "Address.h"
#include "language/SIL/SILType.h"
#include <vector>

namespace language {
namespace irgen {

class IRGenFunction;

/// A temporary allocation.
class Temporary {
public:
  StackAddress Addr;
  SILType Type;

  void destroy(IRGenFunction &IGF) const;
};

class TemporarySet {
  std::vector<Temporary> Stack;
  bool HasBeenCleared = false;

public:
  TemporarySet() = default;

  TemporarySet(TemporarySet &&) = default;
  TemporarySet &operator=(TemporarySet &&) = default;

  // Make this move-only to reduce chances of double-destroys.  We can't
  // get too strict with this, though, because we may need to destroy
  // the same set of temporaries along multiple control-flow paths.
  TemporarySet(const TemporarySet &) = delete;
  TemporarySet &operator=(const TemporarySet &) = delete;

  void add(Temporary temp) {
    Stack.push_back(temp);
  }

  /// Destroy all the temporaries.
  void destroyAll(IRGenFunction &IGF) const;

  /// Remove all the temporaries from this set.  This does not destroy
  /// the temporaries.
  void clear() {
    assert(!HasBeenCleared && "already cleared");
    HasBeenCleared = true;
    Stack.clear();
  }

  /// Has clear() been called on this set?
  bool hasBeenCleared() const {
    return HasBeenCleared;
  }
};

} // end namespace irgen
} // end namespace language

#endif
