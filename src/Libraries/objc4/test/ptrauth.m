/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 24, 2024.
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
#include "test.h"

#include <objc/NSObject.h>
#include <objc/runtime.h>

#include <stdio.h>

int count = 0;

@interface ParentClass : NSObject

- (void)doSomething;

@end

@implementation ParentClass

- (void)doSomething
{
    if (++count == 1) {
        printf("doSomething\n");
	fflush(stdout);
    }
}

@end

int main()
{
    for (int n = 0; n < 128; ++n) {
        char name[32];
        snprintf(name, sizeof(name), "PtrAuthTest%d", n);

        Class testClass = objc_allocateClassPair([ParentClass class], name, 0);

        // This should work, because the isa pointer will be signed
        id obj = [[testClass alloc] init];
        [obj doSomething];

        // Hacking the isa pointer to an unsigned value should cause a crash
        ((__bridge struct objc_object *)obj)->isa = testClass;
        [obj doSomething];
    }

    fail("should have crashed when attempting to invoke -doSomething");
}
