/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 25, 2025.
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
//  __blockObjectAssign.m
//  testObjects
//
//  Created by Blaine Garst on 2/5/09.
//  Copyright 2009 Apple. All rights reserved.
//

//  TEST_CFLAGS -framework Foundation

// tests whether assigning to a __block id variable works in a reasonable way


#import <Foundation/Foundation.h>
#import <stdio.h>
#import <Block.h>
#import "test.h"


@interface TestObject : NSObject {
    int version;
}
- (void)hi;
@end

int AllocationCounter = 0;
int DellocationCounter = 0;

@implementation TestObject


- (id)init {
    version = AllocationCounter++;
    return self;
}

- (void)hi {
    //printf("hi from %p, #%d\n", self, version);
}

- (void)dealloc {
    //printf("dealloc %p, #%d called\n", self, version);
    ++DellocationCounter;
    [super dealloc];
}


@end

void testFunction(bool doExecute, bool doCopy) {
    __block id a = [[TestObject alloc] init];
    //printf("testing - will execute? %d\n", doExecute);
    void (^changeA)(void) = ^{
        [a hi];
        [a release];
        a = [[TestObject alloc] init];
        [a hi];
    };
    if (doCopy) changeA = [changeA copy];
    if (doExecute) changeA();
    if (doCopy) [changeA release];
    [a release];
    //printf("done with explict releasing, implicit to follow\n");
}

int main() {
    NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];
    testFunction(false, true);
    testFunction(true, true);
    testFunction(false, false);
    testFunction(true, true);
    [pool drain];
    if (DellocationCounter != AllocationCounter) {
        fail("only recovered %d of %d objects", DellocationCounter, AllocationCounter);
    }

    succeed(__FILE__);
}
