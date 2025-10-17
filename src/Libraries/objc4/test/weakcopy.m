/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 14, 2023.
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

// TEST_CFLAGS -fobjc-weak

#include "test.h"

#include "testroot.i"
#include <stdint.h>
#include <string.h>
#include <objc/objc-runtime.h>

@interface Weak : TestRoot {
  @public
    __weak id value;
}
@end
@implementation Weak
@end

Weak *oldObject;
Weak *newObject;

int main()
{
    testonthread(^{
        TestRoot *value;

        PUSH_POOL {
            value = [TestRoot new];
            testassert(value);
            oldObject = [Weak new];
            testassert(oldObject);
            
            oldObject->value = value;
            testassert(oldObject->value == value);
            
            newObject = [oldObject copy];
            testassert(newObject);
            testassert(newObject->value == oldObject->value);
            
            newObject->value = nil;
            testassert(newObject->value == nil);
            testassert(oldObject->value == value);
        } POP_POOL;
        
        testcollect();
        TestRootDealloc = 0;
        RELEASE_VAR(value);
    });

    testcollect();
    testassert(TestRootDealloc);

#if __has_feature(objc_arc_weak)
    testassert(oldObject->value == nil);
#endif
    testassert(newObject->value == nil);

    RELEASE_VAR(newObject);
    RELEASE_VAR(oldObject);

    succeed(__FILE__);
    return 0;
}
