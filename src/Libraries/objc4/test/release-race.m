/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 16, 2024.
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

// TEST_CONFIG MEM=mrc LANGUAGE=objective-c

#import <dispatch/dispatch.h>
#import <objc/NSObject.h>
#import "test.h"

@interface MyEncoder : NSObject
{
    int x;
}
@end

@implementation MyEncoder

- (id)init
{
    x = 1;
    return self;
}

- (void)close
{
    x = 2;
    [self release];
}

- (void)dealloc
{
    // Make sure that release has the appropriate barriers so that we're
    // guaranteed to see the x=2 above.
    testassertequal(x, 2);
    [super dealloc];
}
@end

int main() {
    // The 1 thread worker pool on simulators makes this slow and pointless.
#if !TARGET_OS_SIMULATOR
    for (unsigned long long i = 0; i < 100000000ULL; i++) {
        if (i % 100000 == 0)
            testprintf("%llu\n", i);

        MyEncoder *enc = [MyEncoder new];
        [enc retain];   // For the first dispatch
        [enc retain];   // For the second one
        MyEncoder __unsafe_unretained *enc_weak = enc;
        dispatch_async(dispatch_get_global_queue(0, 0), ^{
            [enc_weak close];
        });
        dispatch_async(dispatch_get_global_queue(0, 0), ^{
            [enc_weak release];
        });
        [enc release]; // Drop top level reference
    }
#endif
    succeed(__FILE__);
}
