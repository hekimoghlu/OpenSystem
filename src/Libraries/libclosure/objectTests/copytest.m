/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 16, 2024.
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
//  copytest.m
//  bocktest
//
//  Created by Blaine Garst on 3/21/08.
//  Copyright 2008 __MyCompanyName__. All rights reserved.
//
// TEST_CFLAGS -framework Foundation

#import <Foundation/Foundation.h>
#import <Block_private.h>
#import "test.h"

int GlobalInt = 0;
void setGlobalInt(int value) { GlobalInt = value; }

int main(int argc __unused, char *argv[]) {
    NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];
    int y = 0;
    // must use x+y to avoid optimization of using a global block
    void (^callSetGlobalInt)(int x) = ^(int x) { setGlobalInt(x + y); };
    // a block be able to be sent a message
    void (^callSetGlobalIntCopy)(int) = [callSetGlobalInt copy];
    if (callSetGlobalIntCopy == callSetGlobalInt) {
        // testwarn("copy looks like: %s", _Block_dump(callSetGlobalIntCopy));
        fail("copy is identical", argv[0]);
    }
    callSetGlobalIntCopy(10);
    if (GlobalInt != 10) {
        fail("copy did not set global int");
    }
    [pool drain];

    succeed(__FILE__);
}
