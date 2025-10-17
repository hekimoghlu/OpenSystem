/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 20, 2025.
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

#include "test.h"
#include "testroot.i"
#include <string.h>
#include <objc/runtime.h>
#include <ptrauth.h>

@interface Fake : TestRoot @end
@implementation Fake @end

int main()
{
    TestRoot *obj = [TestRoot new];
    void *buf = (__bridge void *)(obj);
    *(Class __ptrauth_objc_isa_pointer *)buf = [Fake class];

    testassert(object_getClass(obj) == [Fake class]);
    testassert(object_setClass(obj, [TestRoot class]) == [Fake class]);
    testassert(object_getClass(obj) == [TestRoot class]);
    testassert(object_setClass(nil, [TestRoot class]) == nil);

    testassert(malloc_size(buf) >= sizeof(id));
    memset(buf, 0, malloc_size(buf));
    testassert(object_setClass(obj, [TestRoot class]) == nil);

    testassert(object_getClass(obj) == [TestRoot class]);
    testassert(object_getClass([TestRoot class]) == object_getClass([TestRoot class]));
    testassert(object_getClass(nil) == Nil);

    testassert(0 == strcmp(object_getClassName(obj), "TestRoot"));
    testassert(0 == strcmp(object_getClassName([TestRoot class]), "TestRoot"));
    testassert(0 == strcmp(object_getClassName(nil), "nil"));
    
    testassert(0 == strcmp(class_getName([TestRoot class]), "TestRoot"));
    testassert(0 == strcmp(class_getName(object_getClass([TestRoot class])), "TestRoot"));
    testassert(0 == strcmp(class_getName(nil), "nil"));

    succeed(__FILE__);
}
