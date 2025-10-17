/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 13, 2025.
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

// TEST_CONFIG MEM=mrc,arc LANGUAGE=c,c++,objc,objc++

// DO NOT include anything else here
#include <objc/objc.h>
// DO NOT include anything else here
Class c = Nil;
SEL s;
IMP i;
id o = nil;
BOOL b = YES;
BOOL b2 = NO;
#if !__has_feature(objc_arc)
__strong void *p;
#endif
id __unsafe_unretained u;
#if __has_feature(objc_arc_weak)
id __weak w;
#endif

void fn(void) __unused;
void fn(void) {
    id __autoreleasing a __unused;
}

// check type inference for blocks returning YES and NO (rdar://10118972)
BOOL (^block1)(void) = ^{ return YES; };
BOOL (^block2)(void) = ^{ return NO; };

#include "test.h"

int main()
{
    testassert(YES);
    testassert(!NO);
#if __cplusplus
    testwarn("rdar://12371870 -Wnull-conversion");
    testassert(!(bool)nil);
    testassert(!(bool)Nil);
#else
    testassert(!nil);
    testassert(!Nil);
#endif

#if __has_feature(objc_bool)
    // YES[array] is disallowed for objc just as true[array] is for C++
#else
    // this will fail if YES and NO do not have enough parentheses
    int array[2] = { 888, 999 };
    testassert(NO[array] == 888);
    testassert(YES[array] == 999);
#endif

    succeed(__FILE__);
}
