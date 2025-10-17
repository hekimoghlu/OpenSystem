/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 30, 2025.
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
//  refcounting.m
//
//  Created by Blaine Garst on 3/21/08.
//  Copyright 2008 __MyCompanyName__. All rights reserved.
//
// TEST_CFLAGS -framework Foundation

#import <Foundation/Foundation.h>
#import <Block.h>
#import <Block_private.h>
#import "test.h"

int main() {
    NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];
    int i = 10;
    void (^blockA)(void) = ^ { printf("i is %d\n", i); };

    // make sure we can retain/release it
    for (int i = 0; i < 1000; ++i) {
        [blockA retain];
    }
    for (int i = 0; i < 1000; ++i) {
        [blockA release];
    }
    // smae for a copy
    void (^blockAcopy)(void) = [blockA copy];
    for (int i = 0; i < 1000; ++i) {
        [blockAcopy retain];
    }
    for (int i = 0; i < 1000; ++i) {
        [blockAcopy release];
    }
    [blockAcopy release];
    // now for the other guy
    blockAcopy = Block_copy(blockA);
        
    for (int i = 0; i < 1000; ++i) {
        void (^blockAcopycopy)(void) = Block_copy(blockAcopy);
        if (blockAcopycopy != blockAcopy) {
            fail("copy %p of copy %p wasn't the same!!", (void *)blockAcopycopy, (void *)blockAcopy);
        }
    }
    for (int i = 0; i < 1000; ++i) {
        Block_release(blockAcopy);
    }
    Block_release(blockAcopy);
    [pool drain];

    succeed(__FILE__);
}
