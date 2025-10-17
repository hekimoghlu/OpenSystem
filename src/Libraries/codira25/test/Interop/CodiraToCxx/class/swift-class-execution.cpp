/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 22, 2023.
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

// (1) Onone, no evolution

// RUN: %empty-directory(%t-onone)

// RUN: %target-language-frontend %S/language-class-in-cxx.code -module-name Class -clang-header-expose-decls=all-public -typecheck -verify -emit-clang-header-path %t-onone/class.h -Onone

// RUN: %target-interop-build-clangxx -c %s -I %t-onone -o %t-onone/language-class-execution.o
// RUN: %target-interop-build-language %S/language-class-in-cxx.code -o %t-onone/language-class-execution -Xlinker %t-onone/language-class-execution.o -module-name Class -Xfrontend -entry-point-function-name -Xfrontend languageMain -Onone

// RUN: %target-codesign %t-onone/language-class-execution
// RUN: %target-run %t-onone/language-class-execution | %FileCheck %s --check-prefixes=CHECK,CHECK-ONONE

// (2) O, no evolution

// RUN: %empty-directory(%t-ofast)

// RUN: %target-language-frontend %S/language-class-in-cxx.code -module-name Class -clang-header-expose-decls=all-public -typecheck -verify -emit-clang-header-path %t-ofast/class.h -O

// RUN: %target-interop-build-clangxx -c %s -I %t-ofast -o %t-ofast/language-class-execution.o
// RUN: %target-interop-build-language %S/language-class-in-cxx.code -o %t-ofast/language-class-execution -Xlinker %t-ofast/language-class-execution.o -module-name Class -Xfrontend -entry-point-function-name -Xfrontend languageMain -O

// RUN: %target-codesign %t-ofast/language-class-execution
// RUN: %target-run %t-ofast/language-class-execution | %FileCheck %s --check-prefixes=CHECK,CHECK-OPT

// (3) Onone, evolution

// RUN: %empty-directory(%t-evo-onone)

// RUN: %target-language-frontend %S/language-class-in-cxx.code -module-name Class -clang-header-expose-decls=all-public -enable-library-evolution -typecheck -verify -emit-clang-header-path %t-evo-onone/class.h -Onone

// RUN: %target-interop-build-clangxx -c %s -I %t-evo-onone -o %t-evo-onone/language-class-execution.o
// RUN: %target-interop-build-language %S/language-class-in-cxx.code -o %t-evo-onone/language-class-execution-evo -Xlinker %t-evo-onone/language-class-execution.o -module-name Class -enable-library-evolution -Xfrontend -entry-point-function-name -Xfrontend languageMain -Onone

// RUN: %target-codesign %t-evo-onone/language-class-execution-evo
// RUN: %target-run %t-evo-onone/language-class-execution-evo | %FileCheck %s --check-prefixes=CHECK,CHECK-ONONE

// (4) O, evolution

// RUN: %empty-directory(%t-evo-ofast)

// RUN: %target-language-frontend %S/language-class-in-cxx.code -module-name Class -clang-header-expose-decls=all-public -enable-library-evolution -typecheck -verify -emit-clang-header-path %t-evo-ofast/class.h -O

// RUN: %target-interop-build-clangxx -c %s -I %t-evo-ofast -o %t-evo-ofast/language-class-execution.o
// RUN: %target-interop-build-language %S/language-class-in-cxx.code -o %t-evo-ofast/language-class-execution-evo -Xlinker %t-evo-ofast/language-class-execution.o -module-name Class -enable-library-evolution -Xfrontend -entry-point-function-name -Xfrontend languageMain -O

// RUN: %target-codesign %t-evo-ofast/language-class-execution-evo
// RUN: %target-run %t-evo-ofast/language-class-execution-evo | %FileCheck %s --check-prefixes=CHECK,CHECK-OPT

// REQUIRES: executable_test

#include <assert.h>
#include "class.h"
#include <cstdio>
#include <utility>

extern "C" size_t language_retainCount(void * _Nonnull obj);

size_t getRetainCount(const Class::ClassWithIntField & languageClass) {
  void *p = language::_impl::_impl_RefCountedClass::getOpaquePointer(languageClass);
  return language_retainCount(p);
}

int main() {
  using namespace Class;

  // Ensure that the class is released.
  {
    auto x = returnClassWithIntField();
    assert(getRetainCount(x) == 1);
  }
// CHECK:      init ClassWithIntField
// CHECK-NEXT: destroy ClassWithIntField
    
  {
    auto x = returnClassWithIntField();
    {
      takeClassWithIntField(x);
      assert(getRetainCount(x) == 1);
      auto x2 = passThroughClassWithIntField(x);
      assert(getRetainCount(x) == 2);
      assert(getRetainCount(x2) == 2);
      takeClassWithIntField(x2);
      assert(getRetainCount(x) == 2);
    }
    assert(getRetainCount(x) == 1);
    takeClassWithIntField(x);
  }
// CHECK-NEXT: init ClassWithIntField
// CHECK-NEXT: ClassWithIntField: 0;
// CHECK-NEXT: ClassWithIntField: 42;
// CHECK-NEXT: ClassWithIntField: 42;
// CHECK-NEXT: destroy ClassWithIntField

  {
    auto x = returnClassWithIntField();
    assert(getRetainCount(x) == 1);
    takeClassWithIntFieldInout(x);
    assert(getRetainCount(x) == 1);
    takeClassWithIntField(x);
  }
// CHECK-NEXT: init ClassWithIntField
// CHECK-ONONE-NEXT: init ClassWithIntField
// CHECK-ONONE-NEXT: destroy ClassWithIntField
// CHECK-OPT-NEXT: destroy ClassWithIntField
// CHECK-OPT-NEXT: init ClassWithIntField
// CHECK-NEXT: ClassWithIntField: -11;
// CHECK-NEXT: destroy ClassWithIntField

  {
    auto x = returnClassWithIntField();
    {
      auto x2 = x;
      assert(getRetainCount(x) == 2);
    }
    assert(getRetainCount(x) == 1);
  }
// CHECK-NEXT: init ClassWithIntField
// CHECK-NEXT: destroy ClassWithIntField

  {
    auto x = returnClassWithIntField();
    {
      auto x2 = returnClassWithIntField();
      assert(getRetainCount(x2) == 1);
      assert(getRetainCount(x) == 1);
      x = x2;
      assert(getRetainCount(x) == 2);
    }
    takeClassWithIntField(x);
    assert(getRetainCount(x) == 1);
  }
// CHECK-NEXT: init ClassWithIntField
// CHECK-NEXT: init ClassWithIntField
// CHECK-NEXT: destroy ClassWithIntField
// CHECK-NEXT: ClassWithIntField: 0;
// CHECK-NEXT: destroy ClassWithIntField

  {
    auto x = returnClassWithIntField();
    assert(getRetainCount(x) == 1);
    ClassWithIntField x2(std::move(x));
    // Moving a Codira class in C++ is not consuming.
    assert(getRetainCount(x) == 2);
  }
// CHECK-NEXT: init ClassWithIntField
// CHECK-NEXT: destroy ClassWithIntField

  {
      auto x = returnClassWithIntField();
      auto x2 = returnClassWithIntField();
      assert(getRetainCount(x) == 1);
      assert(getRetainCount(x2) == 1);
      x2 = std::move(x);
      // Moving a Codira class in C++ is not consuming.
      assert(getRetainCount(x) == 2);
      assert(getRetainCount(x2) == 2);
      takeClassWithIntField(x2);
  }
// CHECK-NEXT: init ClassWithIntField
// CHECK-NEXT: init ClassWithIntField
// CHECK-NEXT: destroy ClassWithIntField
// CHECK-NEXT: ClassWithIntField: 0;
// CHECK-NEXT: destroy ClassWithIntField
  return 0;
}
