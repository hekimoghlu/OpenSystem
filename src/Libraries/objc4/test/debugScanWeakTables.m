/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 16, 2022.
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

// TEST_ENV OBJC_DEBUG_SCAN_WEAK_TABLES=YES OBJC_DEBUG_SCAN_WEAK_TABLES_INTERVAL_NANOSECONDS=1000
// TEST_CRASHES
// TEST_CONFIG MEM=mrc
/*
TEST_RUN_OUTPUT
objc\[\d+\]: Starting background scan of weak references.
objc\[\d+\]: Weak reference at 0x[0-9a-fA-F]+ contains 0x[0-9a-fA-F]+, should contain 0x[0-9a-fA-F]+
objc\[\d+\]: HALTED
END
*/

#include "test.h"
#include "testroot.i"

#include <time.h>

int main() {
    id obj = [TestRoot new];
    id weakLoc = nil;

    objc_storeWeak(&weakLoc, obj);
    memset_s(&weakLoc, sizeof(weakLoc), 0x35, sizeof(weakLoc));

    uint64_t startTime = clock_gettime_nsec_np(CLOCK_UPTIME_RAW_APPROX);
    while (clock_gettime_nsec_np(CLOCK_UPTIME_RAW_APPROX) - startTime < 5000000000) {
        sleep(1);
        printf(".\n");
    }

    fail("Should have crashed scanning weakLoc");
}