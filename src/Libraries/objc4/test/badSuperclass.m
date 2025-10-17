/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 5, 2022.
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

// TEST_CRASHES
/* 
TEST_RUN_OUTPUT
objc\[\d+\]: Memory corruption in class list\.
objc\[\d+\]: HALTED
END
*/

#include "test.h"
#include "testroot.i"

@interface Super : TestRoot @end
@implementation Super @end

@interface Sub : Super @end
@implementation Sub @end

int main()
{
    alarm(10);
    
    Class supercls = [Super class];
    Class subcls = [Sub class];
    id subobj __unused = [Sub alloc];

    // Create a cycle in a superclass chain (Sub->supercls == Sub)
    // then attempt to walk that chain. Runtime should halt eventually.
    _objc_flush_caches(supercls);
    ((Class __ptrauth_objc_super_pointer *)(__bridge void *)subcls)[1] = subcls;
#ifdef CACHE_FLUSH
    _objc_flush_caches(supercls);
#else
    [subobj class];
#endif
    
    fail("should have crashed");
}
