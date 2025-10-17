/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 6, 2023.
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

// TEST_CONFIG

#import <objc/runtime.h>
#import <objc/objc-auto.h>
#include "test.h"
#include "testroot.i"

@interface Super : TestRoot @end
@implementation Super @end

@interface Sub : Super { int array[128]; } @end
@implementation Sub @end

#if __has_feature(objc_arc)
#define object_dispose(x) do {} while (0)
#endif

int main()
{
    Super *s;

    s = class_createInstance([Super class], 0);
    testassert(s);
    testassert(object_getClass(s) == [Super class]);
    testassert(malloc_size((__bridge const void *)s) >= class_getInstanceSize([Super class]));

    object_dispose(s);

    s = class_createInstance([Sub class], 0);
    testassert(s);
    testassert(object_getClass(s) == [Sub class]);
    testassert(malloc_size((__bridge const void *)s) >= class_getInstanceSize([Sub class]));

    object_dispose(s);

    s = class_createInstance([Super class], 100);
    testassert(s);
    testassert(object_getClass(s) == [Super class]);
    testassert(malloc_size((__bridge const void *)s) >= class_getInstanceSize([Super class]) + 100);

    object_dispose(s);

    s = class_createInstance([Sub class], 100);
    testassert(s);
    testassert(object_getClass(s) == [Sub class]);
    testassert(malloc_size((__bridge const void *)s) >= class_getInstanceSize([Sub class]) + 100);

    object_dispose(s);

    s = class_createInstance(Nil, 0);
    testassert(!s);

    testassert(TestRootAlloc == 0);

#if __has_feature(objc_arc)
    // ARC version didn't use object_dispose() 
    // and should have called -dealloc on 4 objects
    testassert(TestRootDealloc == 4);
#else
    // MRC version used object_dispose()
    // which doesn't call -dealloc
    testassert(TestRootDealloc == 0);
#endif
    
    succeed(__FILE__);
}
