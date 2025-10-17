/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 21, 2021.
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
 *  structmember.c
 *  testObjects
 *
 *  Created by Blaine Garst on 9/30/08.
 *  Copyright 2008 __MyCompanyName__. All rights reserved.
 */

// TEST_CONFIG

#include <Block.h>
#include <Block_private.h>
#include <stdio.h>
#include "test.h"

int main() {
    struct stuff {
        long int a;
        long int b;
        long int c;
    } localStuff = { 10, 20, 30 };
    int d = 0;
    
    void (^a)(void) = ^ { printf("d is %d", d); };
    void (^b)(void) = ^ { printf("d is %d, localStuff.a is %lu", d, localStuff.a); };

    unsigned long nominalsize = Block_size(b) - Block_size(a);
#if __cplusplus__
    // need copy+dispose helper for C++ structures
    nominalsize += 2*sizeof(void*);
#endif
    if ((Block_size(b) - Block_size(a)) != nominalsize) {
        // testwarn("dump of b is %s", _Block_dump(b));
        fail("sizeof a is %lu, sizeof b is %lu, expected %lu", Block_size(a), Block_size(b), nominalsize);
    }

    succeed(__FILE__);
}


