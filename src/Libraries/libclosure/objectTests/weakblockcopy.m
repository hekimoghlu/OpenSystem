/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 13, 2023.
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
//
//  weakblock.m
//  testObjects
//
//  Created by Blaine Garst on 10/30/08.
//  Copyright 2008 __MyCompanyName__. All rights reserved.
//
// TEST_CFLAGS -framework Foundation
//
// Super basic test - does compiler a) compile and b) call out on assignments

#import <Foundation/Foundation.h>
#import "Block_private.h"
#import <pthread.h>
#import "test.h"

// provide our own version for testing

int GotCalled = 0;

int Errors = 0;

int recovered = 0;

@interface TestObject : NSObject {
}
@end

@implementation TestObject
- (id)retain {
    fail("Whoops, retain called!");
}
- (void)dealloc {
    ++recovered;
    [super dealloc];
}
@end


id (^testCopy(void))(void) {
    // create test object
    TestObject *to = [[TestObject alloc] init];
    __block TestObject *__weak  testObject = to;    // iniitialization does NOT require support function
    //id (^b)(void) = [^{ return testObject; } copy];  // g++ rejects this
    id (^b)(void) = [^id{ return testObject; } copy];
    return b;
}

void *test(void *arg __unused)
{
    NSMutableArray *array = (NSMutableArray *)arg;

    GotCalled = 0;
    for (int i = 0; i < 200; ++i) {
        [array addObject:testCopy()];
    }

    return NULL;
}

int main() {

    NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];
    NSMutableArray *array = [NSMutableArray array];

    pthread_t th;
    pthread_create(&th, NULL, test, array);
    pthread_join(th, NULL);

    for (id (^b)(void) in array) {
        if (b() == nil) {
            fail("whoops, lost a __weak __block id");
        }
    }
#if __has_feature(objc_arc)
#error fixme port this post-deallocation check from GC
    for (id (^b)(void) in array) {
            if (b() != nil) {
                fail("whoops, kept a __weak __block id");
            }
        }
    }
#endif

    [pool drain];

    succeed(__FILE__);
}
