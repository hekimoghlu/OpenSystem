/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 29, 2025.
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

//===------------ ProfileCounter.h - PGO Propfile counter -------*- C++ -*-===//
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
/// \file Declares ProfileCounter, a convenient type for PGO
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_BASIC_PROFILECOUNTER_H
#define LANGUAGE_BASIC_PROFILECOUNTER_H

#include <cassert>
#include <cstdint>

namespace language {
/// A class designed to be smaller than using Optional<uint64_t> for PGO
class ProfileCounter {
private:
  uint64_t count;

public:
  explicit constexpr ProfileCounter() : count(UINT64_MAX) {}
  constexpr ProfileCounter(uint64_t Count) : count(Count) {
    if (Count == UINT64_MAX) {
      count = UINT64_MAX - 1;
    }
  }

  bool hasValue() const { return count != UINT64_MAX; }
  uint64_t getValue() const {
    assert(hasValue());
    return count;
  }
  explicit operator bool() const { return hasValue(); }

  /// Saturating addition of another counter to this one, meaning that overflow
  /// is avoided. If overflow would have happened, this function returns true
  /// and the maximum representable value will be set in this counter.
  bool add_saturating(ProfileCounter other) {
    assert(hasValue() && other.hasValue());

    // Will we go over the max representable value by adding other?
    if (count > ((UINT64_MAX-1) - other.count)) {
      count = UINT64_MAX - 1;
      return true;
    }

    count += other.count;
    return false;
  }
};
} // end namespace language

#endif // LANGUAGE_BASIC_PROFILECOUNTER_H
