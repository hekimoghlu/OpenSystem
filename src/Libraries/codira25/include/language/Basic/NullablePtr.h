/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 27, 2024.
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

//===--- NullablePtr.h - A pointer that allows null -------------*- C++ -*-===//
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
// This file defines and implements the NullablePtr class.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_BASIC_NULLABLEPTR_H
#define LANGUAGE_BASIC_NULLABLEPTR_H

#include <cassert>
#include <cstddef>
#include <type_traits>
#include "toolchain/Support/PointerLikeTypeTraits.h"

namespace language {
/// NullablePtr pointer wrapper - NullablePtr is used for APIs where a
/// potentially-null pointer gets passed around that must be explicitly handled
/// in lots of places.  By putting a wrapper around the null pointer, it makes
/// it more likely that the null pointer case will be handled correctly.
template<class T>
class NullablePtr {
  T *Ptr;
  struct PlaceHolder {};

public:
  NullablePtr(T *P = 0) : Ptr(P) {}

  template<typename OtherT>
  NullablePtr(NullablePtr<OtherT> Other,
              typename std::enable_if<
                std::is_convertible<OtherT *, T *>::value,
                PlaceHolder
              >::type = PlaceHolder()) : Ptr(Other.getPtrOrNull()) {}
  
  bool isNull() const { return Ptr == 0; }
  bool isNonNull() const { return Ptr != 0; }

  /// get - Return the pointer if it is non-null.
  const T *get() const {
    assert(Ptr && "Pointer wasn't checked for null!");
    return Ptr;
  }

  /// get - Return the pointer if it is non-null.
  T *get() {
    assert(Ptr && "Pointer wasn't checked for null!");
    return Ptr;
  }

  T *getPtrOrNull() { return getPtrOr(nullptr); }
  const T *getPtrOrNull() const { return getPtrOr(nullptr); }

  T *getPtrOr(T *defaultValue) { return Ptr ? Ptr : defaultValue; }
  const T *getPtrOr(const T *defaultValue) const {
    return Ptr ? Ptr : defaultValue;
  }

  explicit operator bool() const { return Ptr; }

  bool operator==(const NullablePtr<T> &other) const {
    return other.Ptr == Ptr;
  }

  bool operator!=(const NullablePtr<T> &other) const {
    return !(*this == other);
  }

  bool operator==(const T *other) const { return other == Ptr; }

  bool operator!=(const T *other) const { return !(*this == other); }
};
  
} // end namespace language

namespace toolchain {
template <typename T> struct PointerLikeTypeTraits;
template <typename T> struct PointerLikeTypeTraits<language::NullablePtr<T>> {
public:
  static inline void *getAsVoidPointer(language::NullablePtr<T> ptr) {
    return static_cast<void *>(ptr.getPtrOrNull());
  }
  static inline language::NullablePtr<T> getFromVoidPointer(void *ptr) {
    return language::NullablePtr<T>(static_cast<T*>(ptr));
  }
  enum { NumLowBitsAvailable = PointerLikeTypeTraits<T *>::NumLowBitsAvailable };
};

}

#endif // LANGUAGE_BASIC_NULLABLEPTR_H
