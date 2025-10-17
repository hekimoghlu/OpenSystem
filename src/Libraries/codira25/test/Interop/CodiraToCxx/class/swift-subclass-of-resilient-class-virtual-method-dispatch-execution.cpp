/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 1, 2023.
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

// RUN: %target-language-frontend %S/language-subclass-of-resilient-class-virtual-method-dispatch.code -D RESILIENT_MODULE -module-name Class -emit-module -emit-module-path %t/Class.codemodule -enable-library-evolution -clang-header-expose-decls=all-public -emit-clang-header-path %t/class.h

// RUN: %target-language-frontend %S/language-subclass-of-resilient-class-virtual-method-dispatch.code -I %t -module-name UseClass -clang-header-expose-decls=all-public -typecheck -verify -emit-clang-header-path %t/useclass.h

// RUN: %target-interop-build-clangxx -c %s -I %t -o %t/language-class-execution.o

// RUN: %target-interop-build-language -c %S/language-subclass-of-resilient-class-virtual-method-dispatch.code -D RESILIENT_MODULE -o %t/class.o -module-name Class -enable-library-evolution -Xfrontend -entry-point-function-name -Xfrontend languageMain2

// RUN: %target-interop-build-language %S/language-subclass-of-resilient-class-virtual-method-dispatch.code -I %t -o %t/language-class-execution -Xlinker %t/language-class-execution.o -Xlinker %t/class.o  -module-name UseClass -Xfrontend -entry-point-function-name -Xfrontend languageMain

// RUN: %target-codesign %t/language-class-execution
// RUN: %target-run %t/language-class-execution | %FileCheck %s

// REQUIRES: executable_test

#include "class.h"
#include "useclass.h"
#include <assert.h>

using namespace UseClass;

int main() {
  auto derived = createCrossModuleDerivedClass();
  Class::BaseClass derivedAsBase = derived;
  auto derivedDerived = createCrossModuleDerivedDerivedClass();
  CrossModuleDerivedClass derivedDerivedAsDerived = derivedDerived;

  {
    derived.virtualMethod();
    assert(derived.getComputedProp() == -56);
// CHECK: CrossModuleDerivedClass.virtualMethod
  }

  {
    derived.virtualMethodInDerived();
// CHECK-NEXT: CrossModuleDerivedClass.virtualMethodInDerived
    derivedDerived.virtualMethodInDerived();
// CHECK-NEXT: CrossModuleDerivedDerivedClass.virtualMethodInDerived
    derivedDerivedAsDerived.virtualMethodInDerived();
// CHECK-NEXT: CrossModuleDerivedDerivedClass.virtualMethodInDerived
  }

  {
    derived.virtualMethod2InDerived();
// CHECK-NEXT: CrossModuleDerivedClass.virtualMethod2InDerived
    derivedDerived.virtualMethod2InDerived();
// CHECK-NEXT: CrossModuleDerivedDerivedClass.virtualMethod2InDerived
    derivedDerivedAsDerived.virtualMethod2InDerived();
// CHECK-NEXT: CrossModuleDerivedDerivedClass.virtualMethod2InDerived
  }

  {
    language::Int x;
    x = derived.getDerivedComputedProp();
    assert(x == 23);
    x = derivedDerived.getDerivedComputedProp();
    assert(x == -95);
    x = derivedDerivedAsDerived.getDerivedComputedProp();
    assert(x == -95);
  }
  return 0;
}
