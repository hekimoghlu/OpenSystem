/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 1, 2023.
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

// TEST_CONFIG MEM=mrc

#include "test.h"
#include <objc/NSObject.h>

static int deallocCalls;
static int initiateDeallocCalls;

@interface SuperTest: NSObject
@end

@implementation SuperTest

- (id)init {
    deallocCalls = 0;
    initiateDeallocCalls = 0;
    return self;
}

- (void)dealloc {
    deallocCalls++;
    [super dealloc];
}

- (void)_objc_initiateDealloc {
    initiateDeallocCalls++;
    [super dealloc];
}

@end

@interface SubTestRealizedBeforeSet: SuperTest @end
@implementation SubTestRealizedBeforeSet @end

@interface SubTestRealizedAfterSet: SuperTest @end
@implementation SubTestRealizedAfterSet @end

int main()
{
    // Realize classes and test the normal dealloc path.
    [[SuperTest new] release];
    testassertequal(deallocCalls, 1);
    testassertequal(initiateDeallocCalls, 0);

    [[SubTestRealizedBeforeSet new] release];
    testassertequal(deallocCalls, 1);
    testassertequal(initiateDeallocCalls, 0);

    Class DynamicSubTestCreatedBeforeSet = objc_allocateClassPair(
        [SuperTest class], "DynamicSubTestCreatedBeforeSet", 0);
    objc_registerClassPair(DynamicSubTestCreatedBeforeSet);
    [[DynamicSubTestCreatedBeforeSet new] release];
    testassertequal(deallocCalls, 1);
    testassertequal(initiateDeallocCalls, 0);

    // Set custom dealloc initiation and test the above classes again.
    _class_setCustomDeallocInitiation([SuperTest class]);

    [[SuperTest new] release];
    testassertequal(deallocCalls, 0);
    testassertequal(initiateDeallocCalls, 1);

    [[SubTestRealizedBeforeSet new] release];
    testassertequal(deallocCalls, 0);
    testassertequal(initiateDeallocCalls, 1);

    [[DynamicSubTestCreatedBeforeSet new] release];
    testassertequal(deallocCalls, 0);
    testassertequal(initiateDeallocCalls, 1);

    // Test subclasses that are realized or created after setting custom initiation.
    [[SubTestRealizedAfterSet new] release];
    testassertequal(deallocCalls, 0);
    testassertequal(initiateDeallocCalls, 1);

    Class DynamicSubTestCreatedAfterSet = objc_allocateClassPair(
        [SuperTest class], "DynamicSubTestCreatedAfterSet", 0);
    objc_registerClassPair(DynamicSubTestCreatedAfterSet);
    [[DynamicSubTestCreatedAfterSet new] release];
    testassertequal(deallocCalls, 0);
    testassertequal(initiateDeallocCalls, 1);

    succeed(__FILE__);
}
