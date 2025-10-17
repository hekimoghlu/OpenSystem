/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 15, 2025.
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

// RUN: %target-language-frontend %S/enum-associated-value-class-type-cxx.code -module-name Enums -clang-header-expose-decls=all-public -typecheck -verify -emit-clang-header-path %t/enums.h

// RUN: %target-interop-build-clangxx -c %s -I %t -o %t/language-enums-execution.o
// RUN: %target-interop-build-language %S/enum-associated-value-class-type-cxx.code -o %t/language-enums-execution -Xlinker %t/language-enums-execution.o -module-name Enums -Xfrontend -entry-point-function-name -Xfrontend languageMain

// RUN: %target-codesign %t/language-enums-execution
// RUN: %target-run %t/language-enums-execution

// RUN: %empty-directory(%t-evo)

// RUN: %target-language-frontend %S/enum-associated-value-class-type-cxx.code -module-name Enums -clang-header-expose-decls=all-public -enable-library-evolution -typecheck -verify -emit-clang-header-path %t-evo/enums.h

// RUN: %target-interop-build-clangxx -c %s -I %t-evo -o %t-evo/language-enums-execution.o
// RUN: %target-interop-build-language %S/enum-associated-value-class-type-cxx.code -o %t-evo/language-enums-execution -Xlinker %t-evo/language-enums-execution.o -module-name Enums -enable-library-evolution -Xfrontend -entry-point-function-name -Xfrontend languageMain

// RUN: %target-codesign %t-evo/language-enums-execution
// RUN: %target-run %t-evo/language-enums-execution

// REQUIRES: executable_test

#include <cassert>
#include "enums.h"

using namespace Enums;

extern "C" size_t language_retainCount(void * _Nonnull obj);

size_t getRetainCount(const C & obj) {
  void *p = language::_impl::_impl_RefCountedClass::getOpaquePointer(obj);
  return language_retainCount(p);
}

int main() {
    auto c = C::init(1234);
    assert(c.getX() == 1234);
    assert(getRetainCount(c) == 1);

    {
        auto e = E::c(c);
        assert(e.isC());
        assert(getRetainCount(c) == 2);

        auto extracted = e.getC();
        assert(getRetainCount(c) == 3);
        assert(getRetainCount(extracted) == 3);
        assert(extracted.getX() == 1234);

        extracted.setX(5678);
        assert(extracted.getX() == 5678);
        assert(c.getX() == 5678);
    }

    {
      auto c = C::init(5);
      auto arr = language::Array<C>::init(c, 2);
      auto f = F::b(arr);
      assert(f.getB().getCount() == 2);
    }

    {
      auto arr = language::Array<language::Int>::init(42, 2);
      auto g = G<language::Int>::b(arr);
      assert(g.getB().getCount() == 2);
    }

    assert(getRetainCount(c) == 1);
    return 0;
}
