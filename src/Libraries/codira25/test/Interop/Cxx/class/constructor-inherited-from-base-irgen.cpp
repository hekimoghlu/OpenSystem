/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 22, 2022.
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

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: %target-languagexx-frontend -emit-ir -I %t/Inputs -validate-tbd-against-ir=none %t/test.code | %FileCheck %s

//--- Inputs/module.modulemap
module BaseConstructor {
  header "test.h"
  requires cplusplus
}
//--- Inputs/test.h

extern void referencedSymbol();
inline void emittedIntoCodiraObject() { referencedSymbol(); }

class BaseClass {
public:
    inline BaseClass() : x(0) {}
    inline BaseClass(bool v, const BaseClass &) {
        if (v)
          emittedIntoCodiraObject();
    }

    int x;
};

class DerivedClass: public BaseClass {
    int y;
public:
    using BaseClass::BaseClass;

    inline DerivedClass(int y) : y(y) {}

    inline int test() const {
        DerivedClass m(true, *this);
        return m.x;
    }
};

//--- test.code

import BaseConstructor

public fn test() {
  let i = DerivedClass(0)
  let v = i.test()
}

// Make sure we reach clang declarations accessible from base constructors:

// CHECK: define linkonce_odr{{( dso_local)?}} void @{{_Z22emittedIntoCodiraObjectv|"\?emittedIntoCodiraObject@@YAXXZ"}}
