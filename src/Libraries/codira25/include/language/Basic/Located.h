/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 16, 2024.
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

//===--- Located.h - Source Location and Associated Value ----------*- C++ -*-===//
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
// Provides a currency data type Located<T> that should be used instead
// of std::pair<T, SourceLoc>.
//
//===----------------------------------------------------------------------===//


#ifndef LANGUAGE_BASIC_LOCATED_H
#define LANGUAGE_BASIC_LOCATED_H
#include "language/Basic/Debug.h"
#include "language/Basic/Toolchain.h"
#include "language/Basic/SourceLoc.h"

namespace language {

/// A currency type for keeping track of items which were found in the source code.
/// Several parts of the compiler need to keep track of a `SourceLoc` corresponding
/// to an item, in case they need to report some diagnostics later. For example,
/// the ClangImporter needs to keep track of where imports were originally written.
/// Located makes it easy to do so while making the code more readable, compared to
/// using `std::pair`.
template <typename T>
struct Located {
  /// The main item whose source location is being tracked.
  T Item;

  /// The original source location from which the item was parsed.
  SourceLoc Loc;

  Located() : Item(), Loc() {}

  Located(T Item, SourceLoc loc) : Item(Item), Loc(loc) {}

  LANGUAGE_DEBUG_DUMP;
  void dump(raw_ostream &os) const;
};

template <typename T>
bool operator ==(const Located<T> &lhs, const Located<T> &rhs) {
  return lhs.Item == rhs.Item && lhs.Loc == rhs.Loc;
}

} // end namespace language

namespace toolchain {

template <typename T, typename Enable> struct DenseMapInfo;

template<typename T>
struct DenseMapInfo<language::Located<T>> {

  static inline language::Located<T> getEmptyKey() {
    return language::Located<T>(DenseMapInfo<T>::getEmptyKey(),
                             DenseMapInfo<language::SourceLoc>::getEmptyKey());
  }

  static inline language::Located<T> getTombstoneKey() {
    return language::Located<T>(DenseMapInfo<T>::getTombstoneKey(),
                             DenseMapInfo<language::SourceLoc>::getTombstoneKey());
  }

  static unsigned getHashValue(const language::Located<T> &LocatedVal) {
    return combineHashValue(DenseMapInfo<T>::getHashValue(LocatedVal.Item),
                            DenseMapInfo<language::SourceLoc>::getHashValue(LocatedVal.Loc));
  }

  static bool isEqual(const language::Located<T> &LHS, const language::Located<T> &RHS) {
    return DenseMapInfo<T>::isEqual(LHS.Item, RHS.Item) &&
           DenseMapInfo<T>::isEqual(LHS.Loc, RHS.Loc);
  }
};
} // namespace toolchain

#endif // LANGUAGE_BASIC_LOCATED_H
