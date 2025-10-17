/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 6, 2022.
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

// TEST_CONFIG MEM=mrc OS=!exclavekit

#include "test.h"
#include "testroot.i"

@implementation TestRoot (Loader)
+(void)load 
{
    [[TestRoot new] autorelease];
    testassertequal((int)TestRootAutorelease, 1);
    testassertequal((int)TestRootDealloc, 0);
}
@end

int main()
{
    // +load's autoreleased object should have deallocated
    testassertequal((int)TestRootDealloc, 1);

    [[TestRoot new] autorelease];
    testassertequal((int)TestRootAutorelease, 2);


    objc_autoreleasePoolPop(objc_autoreleasePoolPush());
    [[TestRoot new] autorelease];
    testassertequal((int)TestRootAutorelease, 3);


    testonthread(^{
        [[TestRoot new] autorelease];
        testassertequal((int)TestRootAutorelease, 4);
        testassertequal((int)TestRootDealloc, 1);
    });
    // thread's autoreleased object should have deallocated
    testassertequal((int)TestRootDealloc, 2);


    // Test no-pool autorelease after a pool was pushed and popped.
    // The simplest POOL_SENTINEL check during pop gets this wrong.
    testonthread(^{
        objc_autoreleasePoolPop(objc_autoreleasePoolPush());
        [[TestRoot new] autorelease];
        testassertequal((int)TestRootAutorelease, 5);
        testassertequal((int)TestRootDealloc, 2);
    });
    testassert(TestRootDealloc == 3
);
    succeed(__FILE__);
}
