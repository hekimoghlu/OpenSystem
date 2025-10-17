/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 23, 2023.
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

//===--- Once.cpp - Runtime support for lazy initialization ---------------===//
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

#include "language/Threading/Once.h"
#include "Private.h"
#include "language/Runtime/Debug.h"
#include <type_traits>

using namespace language;

#if LANGUAGE_THREADING_DARWIN

// On macOS and iOS, language_once is implemented using GCD.
// The compiler emits an inline check matching the barrier-free inline fast
// path of dispatch_once(). See CodiraTargetInfo.OnceDonePredicateValue.

#include <dispatch/dispatch.h>
static_assert(std::is_same<language_once_t, dispatch_once_t>::value,
              "language_once_t and dispatch_once_t must stay in sync");

#endif // LANGUAGE_THREADING_DARWIN

// The compiler generates the language_once_t values as word-sized zero-initialized
// variables, so we want to make sure language_once_t isn't larger than the
// platform word or the function below might overwrite something it shouldn't.
static_assert(sizeof(language_once_t) <= sizeof(void*),
              "language_once_t must be no larger than the platform word");

/// Runs the given function with the given context argument exactly once.
/// The predicate argument must point to a global or static variable of static
/// extent of type language_once_t.
void language::language_once(language_once_t *predicate, void (*fn)(void *),
                       void *context) {
  language::once(*predicate, fn, context);
}
