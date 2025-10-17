/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 21, 2025.
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
//  copyproperty.m
//  bocktest
//
//  Created by Blaine Garst on 3/21/08.
//  Copyright 2008 __MyCompanyName__. All rights reserved.
//
// TEST_CFLAGS -framework Foundation

#import <Foundation/Foundation.h>
#import <stdio.h>
#import "test.h"

@interface TestObject : NSObject {
    long (^getInt)(void);
}
@property(copy) long (^getInt)(void);
@end

@implementation TestObject
@synthesize getInt;
@end

@interface CountingObject : NSObject
@end

int Retained = 0;

@implementation CountingObject
- (id) retain {
    Retained = 1;
    return [super retain];
}
@end

int main() {
    NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];
    TestObject *to = [[TestObject alloc] init];
    CountingObject *co = [[CountingObject alloc] init];
    long (^localBlock)(void) = ^{ return 10L + (long)[co self]; };
    to.getInt = localBlock;    
    if (localBlock == to.getInt) {
        fail("block property not copied!!");
    }
    if (Retained == 0) {
        fail("didn't copy block import");
    }

    [pool drain];

    succeed(__FILE__);
}
