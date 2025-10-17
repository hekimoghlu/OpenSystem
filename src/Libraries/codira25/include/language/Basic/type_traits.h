/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 8, 2022.
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

//===--- type_traits.h - Type traits ----------------------------*- C++ -*-===//
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

#ifndef LANGUAGE_BASIC_TYPETRAITS_H
#define LANGUAGE_BASIC_TYPETRAITS_H

#include <type_traits>
#include "language/Basic/Compiler.h"

#ifndef __has_keyword
#define __has_keyword(__x) !(__is_identifier(__x))
#endif

#ifndef __has_feature
#define LANGUAGE_DEFINED_HAS_FEATURE
#define __has_feature(x) 0
#endif

namespace language {

/// Same as \c std::is_trivially_copyable, which we cannot use directly
/// because it is not implemented yet in all C++11 standard libraries.
///
/// Unlike \c toolchain::isPodLike, this trait should produce a precise result and
/// is not intended to be specialized.
template<typename T>
struct IsTriviallyCopyable {
#if defined(_LIBCPP_VERSION) || LANGUAGE_COMPILER_IS_MSVC
  // libc++ and MSVC implement is_trivially_copyable.
  static const bool value = std::is_trivially_copyable<T>::value;
#elif __has_feature(is_trivially_copyable) || __GNUC__ >= 5
  static const bool value = __is_trivially_copyable(T);
#else
#  error "Not implemented"
#endif
};

template<typename T>
struct IsTriviallyConstructible {
#if defined(_LIBCPP_VERSION) || LANGUAGE_COMPILER_IS_MSVC
  // libc++ and MSVC implement is_trivially_constructible.
  static const bool value = std::is_trivially_constructible<T>::value;
#elif __has_feature(is_trivially_constructible) || __has_keyword(__is_trivially_constructible)
  static const bool value = __is_trivially_constructible(T);
#elif __has_feature(has_trivial_constructor) || __GNUC__ >= 5
  static const bool value = __has_trivial_constructor(T);
#else
#  error "Not implemented"
#endif
};

template<typename T>
struct IsTriviallyDestructible {
#if defined(_LIBCPP_VERSION) || LANGUAGE_COMPILER_IS_MSVC
  // libc++ and MSVC implement is_trivially_destructible.
  static const bool value = std::is_trivially_destructible<T>::value;
#elif __has_feature(is_trivially_destructible) || __has_keyword(__is_trivially_destructible)
  static const bool value = __is_trivially_destructible(T);
#elif __has_feature(has_trivial_destructor) || __GNUC__ >= 5
  static const bool value = __has_trivial_destructor(T);
#else
#  error "Not implemented"
#endif
};

} // end namespace language

#ifdef LANGUAGE_DEFINED_HAS_FEATURE
#undef __has_feature
#endif

#endif // LANGUAGE_BASIC_TYPETRAITS_H
