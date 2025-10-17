/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 4, 2024.
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
//  importedblockcopy.m
//  testObjects
//
//  Created by Blaine Garst on 10/16/08.
//  Copyright 2008 Apple. All rights reserved.
//

// rdar://6297435
// TEST_CFLAGS -framework Foundation

#import <Foundation/Foundation.h>
#import "Block.h"
#import "test.h"

int Allocated = 0;
int Reclaimed = 0;

@interface TestObject : NSObject
@end

@implementation TestObject
- (void) dealloc {
    ++Reclaimed;
    [super dealloc];
}

- (id)init {
    self = [super init];
    ++Allocated;
    return self;
}

@end

void theTest() {
    // establish a block with an object reference
    TestObject *to = [[TestObject alloc] init];
    void (^inner)(void) = ^ {
        [to self];  // something that will hold onto "to"
    };
    // establish another block that imports the first one...
    void (^outer)(void) = ^ {
        inner();
        inner();
    };
    // now when we copy outer the compiler will _Block_copy_assign inner
    void (^outerCopy)(void) = Block_copy(outer);
    // but when released, at least under GC, it won't let go of inner (nor its import: "to")
    Block_release(outerCopy);
    [to release];
}


int main() {
    NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];
    
    for (int i = 0; i < 200; ++i)
        theTest();
    [pool drain];

    if ((Reclaimed+10) <= Allocated) {
        fail("whoops, reclaimed only %d of %d allocated", Reclaimed, Allocated);
    }

    succeed(__FILE__);
}
