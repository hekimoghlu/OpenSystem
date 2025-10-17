/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 9, 2023.
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

// RUN: %target-language-frontend %S/language-actor-in-cxx.code -module-name Actor -clang-header-expose-decls=has-expose-attr -typecheck -verify -emit-clang-header-path %t/actor.h -disable-availability-checking

// RUN: %target-interop-build-clangxx -c %s -I %t -o %t/language-actor-execution.o
// RUN: %target-interop-build-language %S/language-actor-in-cxx.code -o %t/language-actor-execution -Xlinker %t/language-actor-execution.o -module-name Actor -Xfrontend -entry-point-function-name -Xfrontend languageMain -Xfrontend -disable-availability-checking

// RUN: %target-codesign %t/language-actor-execution
// RUN: %target-run %t/language-actor-execution | %FileCheck %s

// RUN: %empty-directory(%t-evo)

// RUN: %target-language-frontend %S/language-actor-in-cxx.code -module-name Actor -clang-header-expose-decls=has-expose-attr -enable-library-evolution -typecheck -verify -emit-clang-header-path %t-evo/actor.h -disable-availability-checking

// RUN: %target-interop-build-clangxx -c %s -I %t-evo -o %t-evo/language-actor-execution.o
// RUN: %target-interop-build-language %S/language-actor-in-cxx.code -o %t-evo/language-actor-execution-evo -Xlinker %t-evo/language-actor-execution.o -module-name Actor -enable-library-evolution -Xfrontend -entry-point-function-name -Xfrontend languageMain -Xfrontend -disable-availability-checking

// RUN: %target-codesign %t-evo/language-actor-execution-evo
// RUN: %target-run %t-evo/language-actor-execution-evo | %FileCheck %s

// REQUIRES: executable_test

// REQUIRES: concurrency

#include <assert.h>
#include "actor.h"
#include <cstdio>

extern "C" size_t language_retainCount(void * _Nonnull obj);

size_t getRetainCount(const Actor::ActorWithField & languageClass) {
  void *p = language::_impl::_impl_RefCountedClass::getOpaquePointer(languageClass);
  return language_retainCount(p);
}

int main() {
  using namespace Actor;

  // Ensure that the actor class is released.
  {
    auto x = ActorWithField::init();
    assert(getRetainCount(x) == 1);
  }
// CHECK:      init ActorWithField
// CHECK-NEXT: destroy ActorWithField
    
  {
    auto x = ActorWithField::init();
    {
      takeActorWithIntField(x);
      assert(getRetainCount(x) == 1);
    }
    assert(getRetainCount(x) == 1);
  }
// CHECK-NEXT: init ActorWithField
// CHECK-NEXT: takeActorWithIntField
// CHECK-NEXT: destroy ActorWithField
    
  {
    auto x = ActorWithField::init();
    x.method();
  }
// CHECK-NEXT: init ActorWithField
// CHECK-NEXT: nonisolated method
// CHECK-NEXT: destroy ActorWithField
  return 0;
}
