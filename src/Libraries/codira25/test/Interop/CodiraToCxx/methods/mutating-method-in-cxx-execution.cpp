/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 12, 2022.
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

// RUN: %target-language-frontend %S/mutating-method-in-cxx.code -module-name Methods -clang-header-expose-decls=all-public -typecheck -verify -emit-clang-header-path %t/methods.h

// RUN: %target-interop-build-clangxx -c %s -I %t -o %t/language-methods-execution.o
// RUN: %target-interop-build-language %S/mutating-method-in-cxx.code -o %t/language-methods-execution -Xlinker %t/language-methods-execution.o -module-name Methods -Xfrontend -entry-point-function-name -Xfrontend languageMain

// RUN: %target-codesign %t/language-methods-execution
// RUN: %target-run %t/language-methods-execution | %FileCheck %s

// REQUIRES: executable_test

#include <assert.h>
#include "methods.h"

int main() {
  using namespace Methods;

  auto smallStruct = createSmallStruct(10.0f);
  smallStruct.dump();
// CHECK: small x = 10.0;

  smallStruct.scale(0.25f).dump();
  smallStruct.dump();
// CHECK-NEXT: small x = 10.0;
// CHECK-NEXT: small x = 2.5;

  smallStruct.invert();
  smallStruct.dump();
// CHECK-NEXT: small x = -2.5;

  auto largeStruct = createLargeStruct();
  largeStruct.dump();
// CHECK-NEXT: 1, -5, 9, 11, 48879, -77

  largeStruct.double_();
  largeStruct.dump();
// CHECK-NEXT: 2, -10, 18, 22, 97758, -154

  largeStruct.scale(-3, 10).dump();
  largeStruct.dump();
// CHECK-NEXT: -6, -100, -54, 220, -293274, -1540
// CHECK-NEXT: -6, -100, -54, 220, -293274, -1540
  return 0;
}
