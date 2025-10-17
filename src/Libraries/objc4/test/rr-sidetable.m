/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 6, 2022.
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

// TEST_CONFIG OS=!exclavekit MEM=mrc ARCH=x86_64
// TEST_CFLAGS -framework Foundation

// Stress-test nonpointer isa's side table retain count transfers.

// x86_64 only. arm64's side table limit is high enough that bugs 
// are harder to reproduce.

#include "test.h"
#import <Foundation/Foundation.h>

#define OBJECTS 10
#define LOOPS 256
#define THREADS 16
#if __x86_64__
#   define RC_HALF  (1ULL<<7)
#else
#   error sorry
#endif
#define RC_DELTA RC_HALF

static bool Deallocated = false;
@interface Deallocator : NSObject @end
@implementation Deallocator
-(void)dealloc {
    Deallocated = true;
    [super dealloc];
}
@end

// This is global to avoid extra retains by the dispatch block objects.
static Deallocator *obj;

int main() {
    dispatch_queue_t queue = 
        dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);

    for (size_t i = 0; i < OBJECTS; i++) {
        obj = [Deallocator new];

        dispatch_apply(THREADS, queue, ^(size_t i __unused) {
            for (size_t a = 0; a < LOOPS; a++) {
                for (size_t b = 0; b < RC_DELTA; b++) {
                    [obj retain];
                }
                for (size_t b = 0; b < RC_DELTA; b++) {
                    [obj release];
                }
            }
        });

        testassert(!Deallocated);
        [obj release];
        testassert(Deallocated);
        Deallocated = false;
    }

    succeed(__FILE__);
}
