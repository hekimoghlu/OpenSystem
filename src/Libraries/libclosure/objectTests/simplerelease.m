/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 27, 2022.
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
// test -release of block-captured objects

// TEST_CFLAGS -framework Foundation

#import <Foundation/Foundation.h>
#import "Block_private.h"
#import "test.h"

int global = 0;

@interface TestObject : NSObject
@end
@implementation TestObject
- (oneway void)release {
    global = 1;
    [super release];
}
@end

int main() {
    NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];

    TestObject *to = [[TestObject alloc] init];
    void (^b)(void) = ^{ printf("to is at %p\n", to); abort(); };

    // verify that b has a copy/dispose helper
    struct Block_layout *layout = (struct Block_layout *)(void *)b;
    if (!(layout->flags & BLOCK_HAS_COPY_DISPOSE)) {
        fail("Whoops, no copy dispose!");
    }

    _Block_get_dispose_function(layout)(layout);

    if (global != 1) {
       fail("Whoops, helper routine didn't release captive object");
    }
    [pool drain];

    succeed(__FILE__);
}
