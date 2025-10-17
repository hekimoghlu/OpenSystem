/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 6, 2024.
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

// TEST_CONFIG OS=!exclavekit
// TEST_CFLAGS -Os -framework Foundation
// TEST_DISABLED pending clang support for rdar://20530049

#include "test.h"
#include "testroot.i"

#include <objc/objc-internal.h>
#include <objc/objc-abi.h>
#include <Foundation/Foundation.h>

@interface TestObject : TestRoot @end
@implementation TestObject @end

@interface Tester : NSObject @end
@implementation Tester {
@public
    id ivar;
}

-(id) return0 {
    return ivar;
}
-(id) return1 {
    id x = ivar;
    [x self];
    return x;
}

@end

OBJC_EXPORT
id
objc_retainAutoreleasedReturnValue(id obj);

// Accept a value returned through a +0 autoreleasing convention for use at +0.
OBJC_EXPORT
id
objc_unsafeClaimAutoreleasedReturnValue(id obj);


int
main()
{
    TestObject *obj;
    Tester *tt = [Tester new];

#ifdef __x86_64__
    // need to get DYLD to resolve the stubs on x86
    PUSH_POOL {
        TestObject *warm_up = [[TestObject alloc] init];
        testassert(warm_up);
        warm_up = objc_retainAutoreleasedReturnValue(warm_up);
        warm_up = objc_unsafeClaimAutoreleasedReturnValue(warm_up);
        warm_up = nil;
    } POP_POOL;
#endif
    
    testprintf("  Successful +1 -> +1 handshake\n");
    
    PUSH_POOL {
        obj = [[TestObject alloc] init];
        testassert(obj);
        tt->ivar = obj;
        obj = nil;
        
        TestRootRetain = 0;
        TestRootRelease = 0;
        TestRootAutorelease = 0;
        TestRootDealloc = 0;

        TestObject *tmp = [tt return1];

        testassert(TestRootDealloc == 0);
        testassert(TestRootRetain == 1);
        testassert(TestRootRelease == 0);
        testassert(TestRootAutorelease == 0);

        tt->ivar = nil;
        testassert(TestRootDealloc == 0);
        testassert(TestRootRetain == 1);
        testassert(TestRootRelease == 1);
        testassert(TestRootAutorelease == 0);

        tmp = nil;
        testassert(TestRootDealloc == 1);
        testassert(TestRootRetain == 1);
        testassert(TestRootRelease == 2);
        testassert(TestRootAutorelease == 0);

    } POP_POOL;
    
    testprintf("  Successful +0 -> +0 handshake\n");
    
    PUSH_POOL {
        obj = [[TestObject alloc] init];
        testassert(obj);
        tt->ivar = obj;
        obj = nil;
        
        TestRootRetain = 0;
        TestRootRelease = 0;
        TestRootAutorelease = 0;
        TestRootDealloc = 0;

        __unsafe_unretained TestObject *tmp = [tt return0];

        testassert(TestRootDealloc == 0);
        testassert(TestRootRetain == 0);
        testassert(TestRootRelease == 0);
        testassert(TestRootAutorelease == 0);

        tmp = nil;
        testassert(TestRootDealloc == 0);
        testassert(TestRootRetain == 0);
        testassert(TestRootRelease == 0);
        testassert(TestRootAutorelease == 0);

        tt->ivar = nil;
        testassert(TestRootDealloc == 1);
        testassert(TestRootRetain == 0);
        testassert(TestRootRelease == 1);
        testassert(TestRootAutorelease == 0);

    } POP_POOL;

    
    testprintf("  Successful +1 -> +0 handshake\n");
    
    PUSH_POOL {
        obj = [[TestObject alloc] init];
        testassert(obj);
        tt->ivar = obj;
        obj = nil;
        
        TestRootRetain = 0;
        TestRootRelease = 0;
        TestRootAutorelease = 0;
        TestRootDealloc = 0;

        __unsafe_unretained TestObject *tmp = [tt return1];

        testassert(TestRootDealloc == 0);
        testassert(TestRootRetain == 1);
        testassert(TestRootRelease == 1);
        testassert(TestRootAutorelease == 0);

        tmp = nil;
        testassert(TestRootDealloc == 0);
        testassert(TestRootRetain == 1);
        testassert(TestRootRelease == 1);
        testassert(TestRootAutorelease == 0);

        tt->ivar = nil;
        testassert(TestRootDealloc == 1);
        testassert(TestRootRetain == 1);
        testassert(TestRootRelease == 2);
        testassert(TestRootAutorelease == 0);

    } POP_POOL;


    testprintf("  Successful +0 -> +1 handshake\n");
    
    PUSH_POOL {
        obj = [[TestObject alloc] init];
        testassert(obj);
        tt->ivar = obj;
        obj = nil;
        
        TestRootRetain = 0;
        TestRootRelease = 0;
        TestRootAutorelease = 0;
        TestRootDealloc = 0;

        TestObject *tmp = [tt return0];

        testassert(TestRootDealloc == 0);
        testassert(TestRootRetain == 1);
        testassert(TestRootRelease == 0);
        testassert(TestRootAutorelease == 0);

        tmp = nil;
        testassert(TestRootDealloc == 0);
        testassert(TestRootRetain == 1);
        testassert(TestRootRelease == 1);
        testassert(TestRootAutorelease == 0);

        tt->ivar = nil;
        testassert(TestRootDealloc == 1);
        testassert(TestRootRetain == 1);
        testassert(TestRootRelease == 2);
        testassert(TestRootAutorelease == 0);

    } POP_POOL;

    

    succeed(__FILE__);
    
    return 0;
}
