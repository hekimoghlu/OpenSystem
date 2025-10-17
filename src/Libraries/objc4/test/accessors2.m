/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 3, 2025.
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

// TEST_CONFIG MEM=mrc,arc

#include "test.h"
#include <objc/runtime.h>
#include <objc/objc-abi.h>

#include "testroot.i"

@interface Base : TestRoot {
  @public
    id ivar;
}
@end
@implementation Base @end

int main()
{
    SEL _cmd = @selector(foo);
    Base *o = [Base new];
    ptrdiff_t offset = ivar_getOffset(class_getInstanceVariable([Base class], "ivar"));
    testassert(offset == sizeof(id));

    TestRoot *value = [TestRoot new];

    // fixme test atomicity

    // Original setter API

    testprintf("original nonatomic retain\n");
    o->ivar = nil;
    TestRootRetain = 0;
    TestRootCopyWithZone = 0;
    TestRootMutableCopyWithZone = 0;
    objc_setProperty(o, _cmd, offset, value, NO/*atomic*/, NO/*copy*/);
    testassert(TestRootRetain == 1);
    testassert(TestRootCopyWithZone == 0);
    testassert(TestRootMutableCopyWithZone == 0);
    testassert(o->ivar == value);

    testprintf("original atomic retain\n");
    o->ivar = nil;
    TestRootRetain = 0;
    TestRootCopyWithZone = 0;
    TestRootMutableCopyWithZone = 0;
    objc_setProperty(o, _cmd, offset, value, YES/*atomic*/, NO/*copy*/);
    testassert(TestRootRetain == 1);
    testassert(TestRootCopyWithZone == 0);
    testassert(TestRootMutableCopyWithZone == 0);
    testassert(o->ivar == value);

    testprintf("original nonatomic copy\n");
    o->ivar = nil;
    TestRootRetain = 0;
    TestRootCopyWithZone = 0;
    TestRootMutableCopyWithZone = 0;
    objc_setProperty(o, _cmd, offset, value, NO/*atomic*/, YES/*copy*/);
    testassert(TestRootRetain == 0);
    testassert(TestRootCopyWithZone == 1);
    testassert(TestRootMutableCopyWithZone == 0);
    testassert(o->ivar  &&  o->ivar != value);

    testprintf("original atomic copy\n");
    o->ivar = nil;
    TestRootRetain = 0;
    TestRootCopyWithZone = 0;
    TestRootMutableCopyWithZone = 0;
    objc_setProperty(o, _cmd, offset, value, YES/*atomic*/, YES/*copy*/);
    testassert(TestRootRetain == 0);
    testassert(TestRootCopyWithZone == 1);
    testassert(TestRootMutableCopyWithZone == 0);
    testassert(o->ivar  &&  o->ivar != value);

    testprintf("original nonatomic mutablecopy\n");
    o->ivar = nil;
    TestRootRetain = 0;
    TestRootCopyWithZone = 0;
    TestRootMutableCopyWithZone = 0;
    objc_setProperty(o, _cmd, offset, value, NO/*atomic*/, 2/*copy*/);
    testassert(TestRootRetain == 0);
    testassert(TestRootCopyWithZone == 0);
    testassert(TestRootMutableCopyWithZone == 1);
    testassert(o->ivar  &&  o->ivar != value);

    testprintf("original atomic mutablecopy\n");
    o->ivar = nil;
    TestRootRetain = 0;
    TestRootCopyWithZone = 0;
    TestRootMutableCopyWithZone = 0;
    objc_setProperty(o, _cmd, offset, value, YES/*atomic*/, 2/*copy*/);
    testassert(TestRootRetain == 0);
    testassert(TestRootCopyWithZone == 0);
    testassert(TestRootMutableCopyWithZone == 1);
    testassert(o->ivar  &&  o->ivar != value);


    // Optimized setter API

    testprintf("optimized nonatomic retain\n");
    o->ivar = nil;
    TestRootRetain = 0;
    TestRootCopyWithZone = 0;
    TestRootMutableCopyWithZone = 0;
    objc_setProperty_nonatomic(o, _cmd, value, offset);
    testassert(TestRootRetain == 1);
    testassert(TestRootCopyWithZone == 0);
    testassert(TestRootMutableCopyWithZone == 0);
    testassert(o->ivar == value);

    testprintf("optimized atomic retain\n");
    o->ivar = nil;
    TestRootRetain = 0;
    TestRootCopyWithZone = 0;
    TestRootMutableCopyWithZone = 0;
    objc_setProperty_atomic(o, _cmd, value, offset);
    testassert(TestRootRetain == 1);
    testassert(TestRootCopyWithZone == 0);
    testassert(TestRootMutableCopyWithZone == 0);
    testassert(o->ivar == value);

    testprintf("optimized nonatomic copy\n");
    o->ivar = nil;
    TestRootRetain = 0;
    TestRootCopyWithZone = 0;
    TestRootMutableCopyWithZone = 0;
    objc_setProperty_nonatomic_copy(o, _cmd, value, offset);
    testassert(TestRootRetain == 0);
    testassert(TestRootCopyWithZone == 1);
    testassert(TestRootMutableCopyWithZone == 0);
    testassert(o->ivar  &&  o->ivar != value);

    testprintf("optimized atomic copy\n");
    o->ivar = nil;
    TestRootRetain = 0;
    TestRootCopyWithZone = 0;
    TestRootMutableCopyWithZone = 0;
    objc_setProperty_atomic_copy(o, _cmd, value, offset);
    testassert(TestRootRetain == 0);
    testassert(TestRootCopyWithZone == 1);
    testassert(TestRootMutableCopyWithZone == 0);
    testassert(o->ivar  &&  o->ivar != value);

    succeed(__FILE__);
}
