/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 14, 2023.
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

// RUN: %empty-directory(%t)

// RUN: %target-language-frontend %S/language-impl-defs-in-cxx.code -module-name Core -clang-header-expose-decls=all-public -typecheck -verify -emit-clang-header-path %t/core.h

// RUN: %target-interop-build-clangxx -std=c++17 -c %s -I %t -o %t/language-core-validation.o
// RUN: %target-interop-build-clangxx -std=c++20 -c %s -I %t -o %t/language-core-validation.o
// RUN: %target-interop-build-clangxx -std=c++14 -c %s -I %t -o %t/language-core-validation.o -D SHOULD_FAIL

#include <assert.h>
#include "core.h"

#define CHECK(x) (x)

#ifdef SHOULD_FAIL
#  undef CHECK
#  define CHECK(x) !(x)
#endif

int main() {
  language::_impl::ValueWitnessDestroyTy destroyFn;
  static_assert(CHECK(noexcept(destroyFn(nullptr, nullptr))), "value witness table fns are noexcept");
  return 0;
}
