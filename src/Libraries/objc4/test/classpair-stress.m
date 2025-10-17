/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 20, 2025.
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
#include <objc/NSObject.h>
#include <objc/runtime.h>

int main()
{
    // Allocate a large number of classes and make sure their instances work.
    // This is mostly to ensure that the indexed class system on 32-bit works
    // correctly for the full range of values, and when we run off the end.

    // The indexed class array is currently 32,768 entries. Each iteration will
    // use two (class and metaclass).
    int count = 20000;
    for (int i = 0; i < count; i++) {
        testprintf("Testing iteration %d\n", i);

        char *name;
        asprintf(&name, "TestClass-%d", i);

        Class c = objc_allocateClassPair([NSObject class], name, 0);
        objc_registerClassPair(c);

        testprintf("%s is at %p\n", name, c);

        free(name);

        RELEASE_VALUE([[c alloc] init]);
    }

    succeed(__FILE__);
}
