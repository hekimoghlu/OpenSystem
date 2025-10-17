/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 23, 2024.
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
//  counting.m
//  testObjects
//
//  Created by Blaine Garst on 9/23/08.
//  Copyright 2008 Apple. All rights reserved.
//
// rdar://6557292
// TEST_CFLAGS -framework Foundation

#import <Foundation/Foundation.h>
#import <Block.h>
#import <stdio.h>
#import <libkern/OSAtomic.h>
#import <pthread.h>
#import "test.h"

int allocated = 0;
int recovered = 0;

@interface TestObject : NSObject
@end
@implementation TestObject
- (id)init {
    // printf("allocated...\n");
    OSAtomicIncrement32(&allocated);
    return self;
}
- (void)dealloc {
    // printf("deallocated...\n");
    OSAtomicIncrement32(&recovered);
    [super dealloc];
}

@end

void recoverMemory(const char *caller) {
    if (recovered != allocated) {
        fail("after %s recovered %d vs allocated %d", caller, recovered, allocated);
    }
}

// test that basic refcounting works
void *testsingle(void *arg __unused) {
    TestObject *to = [TestObject new];
    void (^b)(void) = [^{ printf("hi %p\n", to); } copy];
    [b release];
    [to release];
    return NULL;
}

void *testlatch(void *arg __unused) {
    TestObject *to = [TestObject new];
    void (^b)(void) = [^{ printf("hi %p\n", to); } copy];
    for (int i = 0; i < 0xfffff; ++i) {
        (void)Block_copy(b);
    }
    for (int i = 0; i < 10; ++i) {
        Block_release(b);
    }
    [b release];
    [to release];
    // lie - b should not be recovered because it has been over-retained
    OSAtomicIncrement32(&recovered);
    return NULL;
}

void *testmultiple(void *arg __unused) {
    TestObject *to = [TestObject new];
    void (^b)(void) = [^{ printf("hi %p\n", to); } copy];
#if 2
    for (int i = 0; i < 10; ++i) {
        (void)Block_copy(b);
    }
    for (int i = 0; i < 10; ++i) {
        Block_release(b);
    }
#endif
    [b release];
    [to release];
    return NULL;
}

int main() {
    pthread_t th;

    pthread_create(&th, NULL, testsingle, NULL);
    pthread_join(th, NULL);
    pthread_create(&th, NULL, testsingle, NULL);
    pthread_join(th, NULL);
    pthread_create(&th, NULL, testsingle, NULL);
    pthread_join(th, NULL);
    pthread_create(&th, NULL, testsingle, NULL);
    pthread_join(th, NULL);
    recoverMemory("testsingle");

    pthread_create(&th, NULL, testlatch, NULL);
    pthread_join(th, NULL);
    recoverMemory("testlatch");

    pthread_create(&th, NULL, testmultiple, NULL);
    pthread_join(th, NULL);
    recoverMemory("testmultiple");

    succeed(__FILE__);
}
