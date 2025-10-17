/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 18, 2023.
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

// RUN: %target-language-frontend %S/getter-in-cxx.code -module-name Properties -clang-header-expose-decls=all-public -typecheck -verify -emit-clang-header-path %t/properties.h

// RUN: %target-interop-build-clangxx -c %s -I %t -o %t/language-props-execution.o
// RUN: %target-interop-build-language %S/getter-in-cxx.code -o %t/language-props-execution -Xlinker %t/language-props-execution.o -module-name Properties -Xfrontend -entry-point-function-name -Xfrontend languageMain

// RUN: %target-codesign %t/language-props-execution
// RUN: %target-run %t/language-props-execution | %FileCheck %s

// REQUIRES: executable_test

#include <assert.h>
#include "properties.h"

int main() {
  using namespace Properties;

  auto smallStructWithGetter = createSmallStructWithGetter();
  assert(smallStructWithGetter.getStoredInt() == 21);
  assert(smallStructWithGetter.getComputedInt() == 23);

  auto largeStruct = smallStructWithGetter.getLargeStruct();
  assert(largeStruct.getX1() == 46);
  assert(largeStruct.getX2() == 1);
  assert(largeStruct.getX3() == 2);
  assert(largeStruct.getX4() == 3);
  assert(largeStruct.getX5() == 4);
  assert(largeStruct.getX6() == 5);

  auto largeStruct2 = largeStruct.getAnotherLargeStruct();
  assert(largeStruct2.getX1() == 11);
  assert(largeStruct2.getX2() == 42);
  assert(largeStruct2.getX3() == -0xFFF);
  assert(largeStruct2.getX4() == 0xbad);
  assert(largeStruct2.getX5() == 5);
  assert(largeStruct2.getX6() == 0);

  auto firstSmallStruct = largeStruct.getFirstSmallStruct();
  assert(firstSmallStruct.getX() == 65);

  auto smallStruct2 = smallStructWithGetter.getSmallStruct();
  assert(smallStruct2.getStoredInt() == 0xFAE);
  assert(smallStruct2.getComputedInt() == (0xFAE + 2));

  {
    StructWithRefCountStoredProp structWithRefCountStoredProp =
      createStructWithRefCountStoredProp();
    auto anotherOne = structWithRefCountStoredProp.getAnother();
    (void)anotherOne;
  }
// CHECK:      create RefCountedClass 0
// CHECK-NEXT: create RefCountedClass 1
// CHECK-NEXT: destroy RefCountedClass 1
// CHECK-NEXT: destroy RefCountedClass 0

  auto propsInClass = createPropsInClass(-1234);
  assert(propsInClass.getStoredInt() == -1234);
  assert(propsInClass.getComputedInt() == -1235);
  auto smallStructFromClass = propsInClass.getSmallStruct();
  assert(smallStructFromClass.getX() == 1234);

  {
    auto x = LargeStruct::getStaticX();
    assert(x == -402);
    auto smallStruct = LargeStruct::getStaticSmallStruct();
    assert(smallStruct.getX() == 789);
  }
  return 0;
}
