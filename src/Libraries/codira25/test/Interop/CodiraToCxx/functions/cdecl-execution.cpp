/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 30, 2024.
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

// RUN: %target-language-frontend %S/cdecl.code -module-name CdeclFunctions -clang-header-expose-decls=all-public -typecheck -verify -emit-clang-header-path %t/functions.h

// RUN: %target-interop-build-clangxx -c %s -I %t -o %t/cdecl-execution.o
// RUN: %target-interop-build-language %S/cdecl.code -o %t/cdecl-execution -Xlinker %t/cdecl-execution.o -module-name Functions -Xfrontend -entry-point-function-name -Xfrontend languageMain

// RUN: %target-codesign %t/cdecl-execution
// RUN: %target-run %t/cdecl-execution

// REQUIRES: executable_test

#include <cassert>
#include "functions.h"

int main() {
  assert(CdeclFunctions::differentCDeclName(1, 1) == 2);
  return 0;
}
