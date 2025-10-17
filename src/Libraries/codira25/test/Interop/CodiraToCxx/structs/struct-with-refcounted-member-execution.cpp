/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 10, 2022.
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

// RUN: %target-language-frontend %S/struct-with-refcounted-member.code -module-name Structs -clang-header-expose-decls=all-public -typecheck -verify -emit-clang-header-path %t/structs.h

// RUN: %target-interop-build-clangxx -c %s -I %t -o %t/language-structs-execution.o
// RUN: %target-interop-build-language %S/struct-with-refcounted-member.code -o %t/language-structs-execution -Xlinker %t/language-structs-execution.o -module-name Structs -Xfrontend -entry-point-function-name -Xfrontend languageMain

// RUN: %target-codesign %t/language-structs-execution
// RUN: %target-run %t/language-structs-execution | %FileCheck %s

// REQUIRES: executable_test

#include <assert.h>
#include "structs.h"

int main() {
  using namespace Structs;

  // Ensure that the value destructor is called.
  {
    StructWithRefcountedMember value = returnNewStructWithRefcountedMember();
  }
  printBreak(1);
// CHECK:      create RefCountedClass
// CHECK-NEXT: destroy RefCountedClass
// CHECK-NEXT: breakpoint 1

  {
    StructWithRefcountedMember value = returnNewStructWithRefcountedMember();
    StructWithRefcountedMember copyValue(value);
  }
  printBreak(2);
// CHECK-NEXT: create RefCountedClass
// CHECK-NEXT: destroy RefCountedClass
// CHECK-NEXT: breakpoint 2

  {
    StructWithRefcountedMember value = returnNewStructWithRefcountedMember();
    StructWithRefcountedMember value2(returnNewStructWithRefcountedMember());
  }
  printBreak(3);
// CHECK-NEXT: create RefCountedClass
// CHECK-NEXT: create RefCountedClass
// CHECK-NEXT: destroy RefCountedClass
// CHECK-NEXT: destroy RefCountedClass
// CHECK-NEXT: breakpoint 3

  {
    StructWithRefcountedMember value = returnNewStructWithRefcountedMember();
    StructWithRefcountedMember value2 = returnNewStructWithRefcountedMember();
    value = value2;
    printBreak(4);
  }
  printBreak(5);
// CHECK-NEXT: create RefCountedClass
// CHECK-NEXT: create RefCountedClass
// CHECK-NEXT: destroy RefCountedClass
// CHECK-NEXT: breakpoint 4
// CHECK-NEXT: destroy RefCountedClass
// CHECK-NEXT: breakpoint 5
  return 0;
}
