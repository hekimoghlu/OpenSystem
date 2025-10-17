/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 20, 2021.
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

// RUN: %target-language-frontend %S/resilient-enum-in-cxx.code -enable-library-evolution -module-name Enums -clang-header-expose-decls=all-public -typecheck -verify -emit-clang-header-path %t/enums.h

// RUN: %target-interop-build-clangxx -c %s -I %t -o %t/language-enums-execution.o

// RUN: %target-interop-build-language %S/resilient-enum-in-cxx.code -enable-library-evolution -o %t/language-enums-execution -Xlinker %t/language-enums-execution.o -module-name Enums -Xfrontend -entry-point-function-name -Xfrontend languageMain
// RUN: %target-codesign %t/language-enums-execution
// RUN: %target-run %t/language-enums-execution | %FileCheck --check-prefixes=CHECK,OLD_CASE %s

// RUN: %target-interop-build-language %S/resilient-enum-in-cxx.code -enable-library-evolution -o %t//language-enums-execution-new -Xlinker %t/language-enums-execution.o -module-name Enums -Xfrontend -entry-point-function-name -Xfrontend languageMain -D NEW_CASE
// RUN: %target-codesign %t/language-enums-execution-new
// RUN: %target-run %t/language-enums-execution-new | %FileCheck --check-prefixes=CHECK,NEW_CASE %s

// REQUIRES: executable_test

#include <cassert>
#include <iostream>
#include "enums.h"

using namespace Enums;

void useFooInSwitch(const Foo& f) {
  switch (f) {
    case Foo::a:
      std::cout << "Foo::a\n";
      break;;
    case Foo::unknownDefault:
      std::cout << "Foo::unknownDefault\n";
      break;
  }
}

int main() {
  auto f1 = makeFoo(10);
  auto f2 = makeFoo(-10);

  printFoo(f1);
  printFoo(f2);

  assert(!f2.isUnknownDefault());
  if (f1.isUnknownDefault()) {
    std::cout << "f1.inResilientUnknownCase()\n";
    assert(!f1.isA());
  } else {
    assert(f1.isA());
    assert(f1.getA() == 10.0);
  }

  useFooInSwitch(f1);
  useFooInSwitch(f2);

  return 0;
}

// OLD_CASE: a(10.0)
// NEW_CASE: b(10)
// CHECK-NEXT: a(-10.0)

// NEW_CASE: f1.inResilientUnknownCase()

// NEW_CASE: Foo::unknownDefault
// OLD_CASE: Foo::a
// CHECK-NEXT: Foo::a
