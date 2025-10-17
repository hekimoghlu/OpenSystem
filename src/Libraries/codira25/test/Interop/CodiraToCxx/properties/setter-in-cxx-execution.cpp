/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 2, 2025.
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

// RUN: %target-language-frontend %S/setter-in-cxx.code -module-name Properties -clang-header-expose-decls=all-public -typecheck -verify -emit-clang-header-path %t/properties.h

// RUN: %target-interop-build-clangxx -c %s -I %t -o %t/language-props-execution.o
// RUN: %target-interop-build-language %S/setter-in-cxx.code -o %t/language-props-execution -Xlinker %t/language-props-execution.o -module-name Properties -Xfrontend -entry-point-function-name -Xfrontend languageMain

// RUN: %target-codesign %t/language-props-execution
// RUN: %target-run %t/language-props-execution | %FileCheck %s

// REQUIRES: executable_test

#include <assert.h>
#include "properties.h"

int main() {
  using namespace Properties;

  auto smallStructWithProps = createSmallStructWithProps();
  smallStructWithProps.setStoredInt(12);
  assert(smallStructWithProps.getStoredInt() == 12);
  assert(smallStructWithProps.getComputedInt() == 14);
  smallStructWithProps.setComputedInt(45);
  assert(smallStructWithProps.getStoredInt() == 43);
  assert(smallStructWithProps.getComputedInt() == 45);
    
  auto largeStructWithProps = smallStructWithProps.getLargeStructWithProps();
  assert(largeStructWithProps.getStoredSmallStruct().getX() == 0xFAE);
  largeStructWithProps.setStoredSmallStruct(createFirstSmallStruct(999));
  assert(largeStructWithProps.getStoredSmallStruct().getX() == 999);

  auto firstSmallStruct = largeStructWithProps.getStoredSmallStruct();
  assert(firstSmallStruct.getX() == 999);
  firstSmallStruct.setX(42);
  assert(firstSmallStruct.getX() == 42);

  largeStructWithProps.setStoredLargeStruct(largeStructWithProps.getStoredLargeStruct());

  smallStructWithProps.setLargeStructWithProps(largeStructWithProps);
// CHECK: SET: LargeStruct(x1: 90, x2: 1, x3: 2, x4: 3, x5: 4, x6: 5), FirstSmallStruct(x: 999)
    
  auto largeStruct = largeStructWithProps.getStoredLargeStruct();
  largeStruct.setX1(0);
  largeStruct.setX2(largeStruct.getX2() * 2);
  largeStruct.setX3(-72);
  largeStructWithProps.setStoredLargeStruct(largeStruct);

  smallStructWithProps.setLargeStructWithProps(largeStructWithProps);
// CHECK-NEXT: SET: LargeStruct(x1: 0, x2: 2, x3: -72, x4: 3, x5: 4, x6: 5), FirstSmallStruct(x: 999)

  auto propsInClass = createPropsInClass(-1234);
  assert(propsInClass.getStoredInt() == -1234);
  propsInClass.setStoredInt(45);
  assert(propsInClass.getStoredInt() == 45);
  propsInClass.setComputedInt(-11);
  assert(propsInClass.getComputedInt() == -11);
  assert(propsInClass.getStoredInt() == -13);

  {
    auto x = LargeStruct::getStaticX();
    assert(x == 0);
    LargeStruct::setStaticX(13);
    x = LargeStruct::getStaticX();
    assert(x == 13);
  }
  return 0;
}
