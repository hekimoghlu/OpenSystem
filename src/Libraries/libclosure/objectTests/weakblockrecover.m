/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 18, 2022.
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
//  weakblockrecover.m
//  testObjects
//
//  Created by Blaine Garst on 11/3/08.
//  Copyright 2008 __MyCompanyName__. All rights reserved.
//

// TEST_CFLAGS -framework Foundation

// rdar://5847976



#import <Foundation/Foundation.h>
#import <Block.h>
#import "test.h"

int Allocated = 0;
int Recovered = 0;

@interface TestObject : NSObject
@end

@implementation TestObject

- (id)init {
    ++Allocated;
    return self;
}
- (void)dealloc {
    ++Recovered;
    [super dealloc];
}

@end

void testRecovery() {
    NSMutableArray *listOfBlocks = [NSMutableArray new];
    for (int i = 0; i < 1000; ++i) {
        __block TestObject *__weak to = [[TestObject alloc] init];
        void (^block)(void) = ^ { printf("is it still real? %p\n", to); };
        [listOfBlocks addObject:[block copy]];
        [to release];
    }

    [listOfBlocks self]; // by using it here we keep listOfBlocks alive across the GC
}

int main() {
    testRecovery();
    if ((Recovered + 10) < Allocated) {
        fail("Only %d weakly referenced test objects recovered, vs %d allocated\n", Recovered, Allocated);
    }

    succeed(__FILE__);
}
