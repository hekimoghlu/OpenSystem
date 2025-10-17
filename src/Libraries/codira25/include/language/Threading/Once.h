/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 20, 2025.
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

//===--- Once.h - Runtime support for lazy initialization -------*- C++ -*-===//
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
// Codira runtime functions in support of lazy initialization.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_THREADING_ONCE_H
#define LANGUAGE_THREADING_ONCE_H

#include "Impl.h"

namespace language {

using once_t = threading_impl::once_t;

/// Runs the given function with the given context argument exactly once.
/// The predicate argument must refer to a global or static variable of static
/// extent of type language::once_t.
inline void once(once_t &predicate, void (*fn)(void *),
                 void *context = nullptr) {
  threading_impl::once_impl(predicate, fn, context);
}

/// Executes the given callable exactly once.
/// The predicate argument must refer to a global or static variable of static
/// extent of type language::once_t.
template <typename Callable>
inline void once(once_t &predicate, const Callable &callable) {
  once(predicate, [](void *ctx) {
    const Callable &callable = *(const Callable*)(ctx);
    callable();
  }, (void *)(&callable));
}

} // namespace language

#endif // LANGUAGE_THREADING_ONCE_H
