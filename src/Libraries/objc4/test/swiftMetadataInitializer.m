/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 24, 2021.
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
#include "swift-class-def.m"
#include <ptrauth.h>

SWIFT_CLASS(SwiftSuper, NSObject, initSuper);
SWIFT_CLASS(SwiftSub, SwiftSuper, initSub);

// _objc_swiftMetadataInitializer hooks for the fake Swift classes

Class initSuper(Class cls __unused, void *arg __unused)
{
    // This test provokes objc's callback out of superclass order.
    // SwiftSub's init is first. SwiftSuper's init is never called.

    fail("SwiftSuper's init should not have been called");
}

static int SubInits = 0;
Class initSub(Class cls, void *arg)
{
    testprintf("initSub callback\n");
    
    testassert(SubInits == 0);
    SubInits++;
    testassert(arg == nil);
    testassert(0 == strcmp(class_getName(cls), "SwiftSub"));
    testassert(cls == RawSwiftSub);
    testassert(!isRealized(RawSwiftSuper));
    testassert(!isRealized(RawSwiftSub));

    testprintf("initSub beginning _objc_realizeClassFromSwift\n");
    _objc_realizeClassFromSwift(cls, cls);
    testprintf("initSub finished  _objc_realizeClassFromSwift\n");

    testassert(isRealized(RawSwiftSuper));
    testassert(isRealized(RawSwiftSub));
    
    return cls;
}


int main()
{
    testassert(SubInits == 0);
    testprintf("calling [SwiftSub class]\n");
    [SwiftSub class];
    testprintf("finished [SwiftSub class]\n");
    testassert(SubInits == 1);
    [SwiftSuper class];
    succeed(__FILE__);
}
