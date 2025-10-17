/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 10, 2025.
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

// RUN: %target-language-frontend %S/large-enums-pass-return-in-cxx.code -module-name Enums -clang-header-expose-decls=all-public -typecheck -verify -emit-clang-header-path %t/enums.h

// RUN: %target-interop-build-clangxx -c %s -I %t -o %t/language-enums-execution.o
// RUN: %target-interop-build-language %S/large-enums-pass-return-in-cxx.code -o %t/language-enums-execution -Xlinker %t/language-enums-execution.o -module-name Enums -Xfrontend -entry-point-function-name -Xfrontend languageMain

// RUN: %target-codesign %t/language-enums-execution
// RUN: %target-run %t/language-enums-execution | %FileCheck %s

// REQUIRES: executable_test

#include <cassert>
#include <cstdint>
#include "enums.h"

int main() {
    using namespace Enums;

    // sizeof(generated cxx class) = 1 + max(sizeof(case) for all cases) + padding
    static_assert(sizeof(Large) == 56, "MemoryLayout<Large>.stride == 56");

    auto large = makeLarge(-1);
    printLarge(large);
    // CHECK: Large.second
    inoutLarge(large, 10);
    printLarge(large);
    // CHECK: Large.first(-1, -2, -3, -4, -5, -6)
    printLarge(passThroughLarge(large));
    // CHECK: Large.first(-1, -2, -3, -4, -5, -6)

    return 0;
}
