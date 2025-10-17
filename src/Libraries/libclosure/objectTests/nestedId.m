/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 22, 2023.
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
//  nestedBlock.m
//  testObjects
//
//  Created by Blaine Garst on 6/24/08.
//  Copyright 2008 __MyCompanyName__. All rights reserved.
//

// test -retain and -release

// TEST_CFLAGS -framework Foundation

#import <stdio.h>
#import <Block.h>
#import <Foundation/Foundation.h>
#import "test.h"

int Retained = 0;

void (^savedBlock)(void);
void (^savedBlock2)(void);

void saveit(void (^block)(void)) {
    savedBlock = Block_copy(block);
}
void callit() {
    savedBlock();
}
void releaseit() {
    Block_release(savedBlock);
    savedBlock = nil;
}
void saveit2(void (^block)(void)) {
    savedBlock2 = Block_copy(block);
}
void callit2() {
    savedBlock2();
}
void releaseit2() {
    Block_release(savedBlock2);
    savedBlock2 = nil;
}

@interface TestObject : NSObject
@end

@implementation TestObject
- (id)retain {
    ++Retained;
    [super retain];
    return self;
}
- (oneway void)release {
    --Retained;
    [super retain];
}

        
@end
id global;

void test(id param) {
    saveit(^{
        saveit2(
            ^{ 
                global = param;
            });
    });
}


int main() {
    TestObject *to = [[TestObject alloc] init];
    
    test(to);
    if (Retained == 0) {
        fail("didn't update Retained");
    }
    callit();
    callit2();

    succeed(__FILE__);
}
