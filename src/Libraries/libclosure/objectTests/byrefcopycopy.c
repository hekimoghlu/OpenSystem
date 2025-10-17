/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 21, 2023.
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

// rdar://6255170

#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <Block.h>
#include <Block_private.h>
#include <assert.h>
#include "test.h"

int main()
{
    __block int var = 0;
    int shouldbe = 0;
    void (^b)(void) = ^{ var++; /*printf("var is at %p with value %d\n", &var, var);*/ };
    __typeof(b) _b;
    //printf("before copy...\n");
    b(); ++shouldbe;
    size_t i;

    for (i = 0; i < 10; i++) {
            _b = Block_copy(b); // make a new copy each time
            assert(_b);
            ++shouldbe;
            _b();               // should still update the stack
            Block_release(_b);
    }

    //printf("after...\n");
    b(); ++shouldbe;

    if (var != shouldbe) {
        fail("var is %d but should be %d", var, shouldbe);
    }

    succeed(__FILE__);
}
