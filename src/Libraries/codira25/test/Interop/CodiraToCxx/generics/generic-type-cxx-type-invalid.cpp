/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 3, 2025.
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

// RUN: %target-language-frontend %S/generic-struct-in-cxx.code -module-name Structs -clang-header-expose-decls=all-public -typecheck -verify -emit-clang-header-path %t/structs.h

// RUN: %target-interop-build-clangxx -std=gnu++20 -c %s -I %t -o /dev/null -DUSE_TYPE=int
// RUN: %target-interop-build-clangxx -std=gnu++17 -c %s -I %t -o /dev/null -DUSE_TYPE=int
// RUN: not %target-interop-build-clangxx -std=gnu++20 -c %s -I %t -o /dev/null -DUSE_TYPE=CxxStruct
// RUN: not %target-interop-build-clangxx -std=gnu++17 -c %s -I %t -o /dev/null -DUSE_TYPE=CxxStruct 2>&1 | %FileCheck -check-prefix=STATIC_ASSERT_ERROR %s

#include "structs.h"

struct CxxStruct { int x = 0; };

bool test(Structs::GenericPair<USE_TYPE, USE_TYPE> &val) {
  return sizeof(val) > 4;
}

// STATIC_ASSERT_ERROR: type cannot be used in a Codira generic context
