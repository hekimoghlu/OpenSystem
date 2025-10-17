/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 18, 2025.
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

// RUN: %target-language-frontend %S/large-structs-pass-return-indirect-in-c.code -module-name Structs -clang-header-expose-decls=all-public -typecheck -verify -emit-clang-header-path %t/structs.h

// RUN: %target-interop-build-clang -c %s -I %t -o %t/language-structs-execution.o
// RUN: %target-interop-build-language %S/large-structs-pass-return-indirect-in-c.code -o %t/language-structs-execution -Xlinker %t/language-structs-execution.o -module-name Structs -Xfrontend -entry-point-function-name -Xfrontend languageMain

// RUN: %target-codesign %t/language-structs-execution
// RUN: %target-run %t/language-structs-execution | %FileCheck %s

// REQUIRES: executable_test

#include <assert.h>
#include "structs.h"

int main() {
  // printStructSeveralI64(returnNewStructSeveralI64(42))
  struct Structs_StructSeveralI64 structSeveralI64;
  $s7Structs25returnNewStructSeveralI641iAA0deF0Vs5Int64V_tF(&structSeveralI64, 42);
  $s7Structs21printStructSeveralI64yyAA0cdE0VF(&structSeveralI64);
// CHECK: StructSeveralI64.1 = 42, .2 = 0, .3 = -17, .4 = 12345612, .5 = -65535

  // printStructSeveralI64(passThroughStructSeveralI64(581, returnNewStructSeveralI64(42), 5.0))
  struct Structs_StructSeveralI64 structSeveralI64_copy;
  $s7Structs27passThroughStructSeveralI641i_1jAA0deF0Vs5Int64V_AFSftF(&structSeveralI64_copy,
    581, &structSeveralI64, 5.0f);
  $s7Structs21printStructSeveralI64yyAA0cdE0VF(&structSeveralI64_copy);
// CHECK-NEXT: StructSeveralI64.1 = 42, .2 = 581, .3 = -17, .4 = -12345612, .5 = -65530
  return 0;
}
