/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 10, 2024.
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
//  notcopied.m
//  testObjects
//
//  Created by Blaine Garst on 2/12/09.
//  Copyright 2009 Apple. All rights reserved.
//

// TEST_CFLAGS -framework Foundation

// rdar://6557292
// Test that a __block Block variable with a reference to a stack based Block is not copied
// when a Block referencing the __block Block varible is copied.
// No magic for __block variables.

#import <stdio.h>
#import <Block.h>
#import <Block_private.h>
#import <Foundation/Foundation.h>
#import "test.h"

int Retained = 0;

@interface TestObject : NSObject
@end
@implementation TestObject
- (id)retain {
    Retained = 1;
    return [super retain];
}
@end


int main() {
    TestObject *to = [[TestObject alloc] init];
    __block void (^someBlock)(void) = ^ { [to self]; };
    void (^someOtherBlock)(void) = ^ {
          someBlock();   // reference someBlock.  It shouldn't be copied under the new rules.
    };
    someOtherBlock = [someOtherBlock copy];
    if (Retained != 0) {
        fail("__block Block was copied when it shouldn't have");
    }

    succeed(__FILE__);
}
