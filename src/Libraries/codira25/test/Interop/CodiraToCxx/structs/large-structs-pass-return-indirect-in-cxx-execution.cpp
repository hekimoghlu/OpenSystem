/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 5, 2025.
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

// RUN: %target-language-frontend %S/large-structs-pass-return-indirect-in-cxx.code -module-name Structs -clang-header-expose-decls=all-public -typecheck -verify -emit-clang-header-path %t/structs.h

// RUN: %target-interop-build-clangxx -c %s -I %t -o %t/language-structs-execution.o
// RUN: %target-interop-build-language %S/large-structs-pass-return-indirect-in-cxx.code -o %t/language-structs-execution -Xlinker %t/language-structs-execution.o -module-name Structs -Xfrontend -entry-point-function-name -Xfrontend languageMain

// RUN: %target-codesign %t/language-structs-execution
// RUN: %target-run %t/language-structs-execution | %FileCheck %s

// REQUIRES: executable_test

#include <assert.h>
#include "structs.h"

int main() {
  using namespace Structs;

  static_assert(sizeof(StructSeveralI64) == 40);

  printStructSeveralI64(returnNewStructSeveralI64(42));
// CHECK: StructSeveralI64.1 = 42, .2 = 0, .3 = -17, .4 = 12345612, .5 = -65535

  StructSeveralI64 structSeveralI64_copy = passThroughStructSeveralI64(100, returnNewStructSeveralI64(11), 6.0);
  printStructSeveralI64(structSeveralI64_copy);
// CHECK-NEXT: StructSeveralI64.1 = 11, .2 = 100, .3 = -17, .4 = -12345612, .5 = -65529

  auto myStruct = returnNewStructSeveralI64(99);
  printStructSeveralI64(myStruct);
// CHECK-NEXT: StructSeveralI64.1 = 99, .2 = 0, .3 = -17, .4 = 12345612, .5 = -65535

  inoutStructSeveralI64(myStruct);
  printStructSeveralI64(myStruct);
// CHECK-NEXT: StructSeveralI64.1 = -1, .2 = -2, .3 = -3, .4 = -4, .5 = -5
  return 0;
}
