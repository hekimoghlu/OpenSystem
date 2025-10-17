/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 21, 2022.
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
//  retainrelease.m
//  bocktest
//
//  Created by Blaine Garst on 3/21/08.
//  Copyright 2008 __MyCompanyName__. All rights reserved.
//
// TEST_CFLAGS -framework Foundation

#import <Foundation/Foundation.h>
#import "test.h"

@interface TestObject : NSObject {
}
@end

int GlobalInt = 0;

@implementation TestObject
- (id) retain {
    ++GlobalInt;
    return self;
}


@end

int main(int argc, char *argv[] __unused) {
   NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];
   // an object should not be retained within a stack Block
   TestObject *to = [[TestObject alloc] init];
   TestObject *to2 = [[TestObject alloc] init];
   void (^blockA)(void) __unused = ^ { [to self]; printf("using argc %d\n", argc); [to2 self]; };
   if (GlobalInt != 0) {
       fail("object retained inside stack closure");
   }
   [pool drain];

   succeed(__FILE__);
}
   
