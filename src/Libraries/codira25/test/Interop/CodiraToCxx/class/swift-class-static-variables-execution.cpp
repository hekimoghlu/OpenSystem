/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 5, 2025.
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

// RUN: %target-language-frontend %S/language-class-static-variables.code -module-name Class -clang-header-expose-decls=all-public -typecheck -verify -emit-clang-header-path %t/class.h

// RUN: %target-interop-build-clangxx -c %s -I %t -o %t/language-class-execution.o
// RUN: %target-interop-build-language %S/language-class-static-variables.code -o %t/language-class-execution -Xlinker %t/language-class-execution.o -module-name Class -Xfrontend -entry-point-function-name -Xfrontend languageMain

// RUN: %target-codesign %t/language-class-execution
// RUN: %target-run %t/language-class-execution

// REQUIRES: executable_test


#include "class.h"
#include <assert.h>
#include <cstdio>

using namespace Class;

int main() {
    auto x = FileUtilities::getShared();
    assert(x.getField() == 42);
}
