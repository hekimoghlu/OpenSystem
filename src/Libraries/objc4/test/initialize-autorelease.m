/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 1, 2025.
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
#include <objc/objc-internal.h>

bool didDealloc;

@interface TestClass: NSObject @end

@implementation TestClass

+ (void)initialize {
    // Verify that autoreleasing an object in +initialize doesn't leak.
    id instance = [[TestClass alloc] init];
    [instance autorelease];
}

- (void)dealloc {
    didDealloc = true;
    [super dealloc];
}

@end

int main()
{
    @autoreleasepool {
        // We need to get to objc_retainAutoreleaseReturnValue without
        // triggering initialization, but we do want it to be realized. Looking
        // up the class by name avoids initialization.
        Class c = objc_getClass("TestClass");

        // Getting the instance size triggers realization if needed.
        class_getInstanceSize(c);

        // Make the call.
        objc_retainAutoreleaseReturnValue(c);
    }
    testassert(didDealloc);

    succeed(__FILE__);
}
