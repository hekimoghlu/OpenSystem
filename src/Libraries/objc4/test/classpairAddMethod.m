/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 6, 2024.
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

// TEST_CONFIG

#include "test.h"

#include "testroot.i"
#include <objc/runtime.h>

void fakeIMP() {}
IMP testIMP = (IMP)fakeIMP;

int main()
{
    // Make sure that method_getDescription doesn't return a stale cached
    // description when creating and destroying dynamic subclasses
    // (rdar://91521212). This is a UaF so it's not completely reliable, but
    // empirically it happens reliably after five iterations. Run 100 just to be
    // safe.
    for (int i = 0; i < 100; i++) {
        char *name;
        asprintf(&name, "test-%d", i);
        SEL sel = sel_getUid(name);
        Class c = objc_allocateClassPair([TestRoot class], name, 0);
        class_addMethod(c, sel, testIMP, name);
        objc_registerClassPair(c);

        Method m = class_getInstanceMethod(c, sel);
        struct objc_method_description *desc = method_getDescription(m);
        testassert(strcmp(desc->types, name) == 0);
        testassertequal(desc->name, sel);

        free(name);
        objc_disposeClassPair(c);
    }

    succeed(__FILE__);
}