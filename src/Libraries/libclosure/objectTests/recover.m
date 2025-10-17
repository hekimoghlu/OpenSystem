/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 15, 2024.
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
// TEST_CFLAGS -framework Foundation

#import <Foundation/Foundation.h>
#import <Block.h>
#import <Block_private.h>
#import "test.h"

int recovered = 0;

@interface TestObject : NSObject {
}
@end
@implementation TestObject
- (void)dealloc {
    ++recovered;
    [super dealloc];
}
@end

typedef struct {
    struct Block_layout layout;  // assumes copy helper
    struct Block_byref *byref_ptr;
} Block_with_byref;

void testRoutine() {
    __block id to = [[TestObject alloc] init];
    void (^b)(void) = [^{ [to self]; } copy];
    for (int i = 0; i < 10; ++i)
        [b retain];
    for (int i = 0; i < 10; ++i)
        [b release];
    for (int i = 0; i < 10; ++i)
        (void)Block_copy(b);            // leak
    for (int i = 0; i < 10; ++i)
        Block_release(b);
    for (int i = 0; i < 10; ++i) {
        (void)Block_copy(b);   // make sure up
        Block_release(b);  // and down work under GC
    }
    [b release];
    [to release];
    // block_byref_release needed under non-GC to get rid of testobject
}
    

int main() {
    NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];
    for (int i = 0; i < 200; ++i)   // do enough to trigger TLC if GC is on
        testRoutine();
    [pool drain];

    if (recovered == 0) {
        fail("didn't recover byref block variable");
    }

    succeed(__FILE__);
}
