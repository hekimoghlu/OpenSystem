/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 20, 2023.
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
//  byrefcopyid.m
//  testObjects
//
//  Created by Blaine Garst on 5/13/08.
//  Copyright 2008 __MyCompanyName__. All rights reserved.
//

// Tests copying of blocks with byref ints and an id
// TEST_CFLAGS -framework Foundation

#import <Foundation/Foundation.h>
#import <Block.h>
#import <Block_private.h>
#import "test.h"


int CalledRetain = 0;
int CalledRelease = 0;
int CalledSelf = 0;
int CalledDealloc = 0;


@interface DumbObject : NSObject {
}
@end

@implementation DumbObject
- (id)retain {
    CalledRetain = 1;
    return [super retain];
}
- (oneway void)release {
    CalledRelease = 1;
    [super release];
}
- (id)self {
    CalledSelf = 1;
    return self;
}

- (void)dealloc {
    CalledDealloc = 1;
    [super dealloc];
}

@end


void callVoidVoid(void (^closure)(void)) {
    closure();
}

void (^dummy)(void);



id testRoutine(const char *whoami) {
    __block id  dumbo = [DumbObject new];
    dummy = ^{
        [dumbo self];
    };
    
    
    //doHack(dummy);
    id copy = Block_copy(dummy);
    
    callVoidVoid(copy);
    if (CalledSelf == 0) {
        fail("copy helper of byref id didn't call self", whoami);
    }

    return copy;
}

int main(int argc __unused, char *argv[]) {
    id copy = testRoutine(argv[0]);
    Block_release(copy);
    if (CalledRetain != 0) {
        fail("copy helper of byref retained the id");
    }
    if (CalledRelease != 0) {
        fail("copy helper of byref id did release the id");
    }
    
    succeed(__FILE__);
}
