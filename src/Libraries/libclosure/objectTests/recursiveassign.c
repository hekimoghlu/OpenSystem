/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 16, 2024.
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
/*
 *  recursiveassign.c
 *  testObjects
 *
 *  Created by Blaine Garst on 12/3/08.
 *  Copyright 2008 __MyCompanyName__. All rights reserved.
 *
 */

// TEST_CONFIG rdar://6639533

// The compiler is prefetching x->forwarding before evaluting code that recomputes forwarding and so the value goes to a place that is never seen again.

#include <stdio.h>
#include <stdlib.h>
#include <Block.h>
#include "test.h"

int main() {
    
    __block void (^recursive_copy_block)(int) = ^(int arg __unused) { 
        fail("got wrong Block"); 
    };
    __block int done = 2;
    
    recursive_copy_block = Block_copy(^(int i) {
        if (i > 0) {
            recursive_copy_block(i - 1);
        }
        else {
            if (done != 0) abort();
            done = 1;
        }
    });
    
    done = 0;
    recursive_copy_block(5);
    testassert(done == 1);
    
    succeed(__FILE__);
}

