/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 4, 2024.
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
#include "test.h"
#include "imageorder.h"
#include <objc/runtime.h>
#include <dlfcn.h>

int main()
{
    // +load methods and C static initializers
    testassert(state == 3);
    testassert(cstate == 3);

    Class cls = objc_getClass("Super");
    testassert(cls);

    // make sure all categories arrived
    state = -1;
    [Super method0];
    testassert(state == 0);
    [Super method1];
    testassert(state == 1);
    [Super method2];
    testassert(state == 2);
    [Super method3];
    testassert(state == 3);

    // make sure imageorder3.dylib is the last category to attach
    state = 0;
    [Super method];
    testassert(state == 3);

    succeed(__FILE__);
}
