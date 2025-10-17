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
//  byrefgc.m
//  testObjects
//
//  Created by Blaine Garst on 5/16/08.
//  Copyright 2008 __MyCompanyName__. All rights reserved.
//

// TEST_CFLAGS -framework Foundation


#import <stdio.h>
#import <Block.h>
#import "test.h"
#import "testroot.i"

int GotHi = 0;

int VersionCounter = 0;

@interface TestObject : TestRoot {
    int version;
}
- (void) hi;
@end

@implementation TestObject


- (id)init {
    version = VersionCounter++;
    return self;
}

- (void) hi {
    GotHi++;
}

@end


void (^get_block(void))(void) {
    __block TestObject * to = [[TestObject alloc] init];
    return [^{ [to hi]; to = [[TestObject alloc] init]; } copy];
}

int main() {
    
    void (^voidvoid)(void) = get_block();
    voidvoid();
    voidvoid();
    voidvoid();
    voidvoid();
    voidvoid();
    voidvoid();
    RELEASE_VAR(voidvoid);
    testprintf("alloc %d dealloc %d\n", TestRootAlloc, TestRootDealloc);
#if __has_feature(objc_arc)
    // one TestObject still alive in get_block's __block variable
    testassert(TestRootAlloc == TestRootDealloc + 1);
#else
    // __block variables are unretained so they all leaked
    testassert(TestRootAlloc == 7);
    testassert(TestRootDealloc == 0);
#endif

    succeed(__FILE__);
}
