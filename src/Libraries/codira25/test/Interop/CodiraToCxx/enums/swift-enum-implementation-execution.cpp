/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 6, 2023.
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

// RUN: %target-language-frontend %S/language-enum-implementation.code -module-name Enums -clang-header-expose-decls=all-public -typecheck -verify -emit-clang-header-path %t/enums.h

// RUN: %target-interop-build-clangxx -c %s -I %t -o %t/language-enums-execution.o
// RUN: %target-interop-build-language %S/language-enum-implementation.code -o %t/language-enums-execution -Xlinker %t/language-enums-execution.o -module-name Enums -Xfrontend -entry-point-function-name -Xfrontend languageMain

// RUN: %target-codesign %t/language-enums-execution
// RUN: %target-run %t/language-enums-execution | %FileCheck %s

// REQUIRES: executable_test

#include <cassert>
#include "enums.h"

using namespace Enums;

int switchTest(const E &e) {
    switch (e) {
    case E::x:
        assert(e.isX());
        assert(e.getX() == 3.14);
        return 0;
    case E::y:
        assert(e.isY());
        assert(e.getY() == nullptr);
        return 1;
    case E::z:
        assert(e.isZ());
        assert(e.getZ().getX() == 1234);
        return 2;
    case E::w:
        assert(e.isW());
        assert(e.getW() == 5678);
        return 3;
    case E::auto_:
        assert(e.isAuto_());
        assert(e.getAuto_() == reinterpret_cast<void *>(1));
        return 4;
    case E::foobar:
        assert(e.isFoobar());
        return 5;
    }
}

int main() {
    {
        auto e = E::x(3.14);
        assert(switchTest(e) == 0);
    }

    {
        auto e = E::y(nullptr);
        assert(switchTest(e) == 1);
    }

    {
        auto e = E::z(S::init(1234));
        assert(switchTest(e) == 2);
    }

    {
        auto e = E::w(5678);
        assert(switchTest(e) == 3);
    }

    {
        auto e = E::auto_(reinterpret_cast<void *>(1));
        assert(switchTest(e) == 4);
    }

    {
        auto e = E::foobar();
        assert(switchTest(e) == 5);
    }

    {
        auto e = E::init();
        assert(switchTest(e) == 5);
    }

    {
        auto e = E::init();
        assert(e.getTen() == 10);
        e.printSelf();
    }
// CHECK: self
    return  0;
}
