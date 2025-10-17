/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 18, 2024.
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

#include <Block.h>
#include <stdio.h>
#include "test.h"

// rdar://6225809
// fixed in 5623

int main() {
    __block int a = 42;
    int* ap = &a; // just to keep the address on the stack.

    void (^b)(void) = ^{
        //a;              // workaround, a should be implicitly imported
        (void)Block_copy(^{
            a = 2;
        });
    };

    (void)Block_copy(b);

    if(&a == ap) {
        fail("__block heap storage should have been created at this point");
    }
    
    succeed(__FILE__);
}
