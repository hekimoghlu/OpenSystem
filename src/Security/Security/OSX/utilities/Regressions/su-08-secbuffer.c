/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 1, 2025.
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
#include <utilities/SecBuffer.h>

#include "utilities_regressions.h"


#define kTestCount (2 * 12 + 3 * 12)

const uint8_t testBytes[] = { 0xD0, 0xD0, 0xBA, 0xAD };

static void
tests(void) {
    for(size_t testSize = 1023; testSize < 2 * 1024 * 1024; testSize *= 2) {
        PerformWithBuffer(testSize, ^(size_t size, uint8_t *buffer) {
            ok(buffer, "got buffer");
            ok(size == testSize, "buffer size");

            // Scribble on the end, make sure we can.
            uint64_t *scribbleLocation = (uint64_t *) (buffer + testSize - sizeof(testBytes));
            bcopy(testBytes, scribbleLocation, sizeof(testBytes));
        });
    }
    
    for(size_t testSize = 1023; testSize < 2 * 1024 * 1024; testSize *= 2) {
        __block uint64_t *scribbleLocation = NULL;
        PerformWithBufferAndClear(testSize, ^(size_t size, uint8_t *buffer) {
            ok(buffer, "got buffer");
            ok(size == testSize, "buffer size");
            
            scribbleLocation = (uint64_t *) (buffer + testSize - sizeof(testBytes));
            bcopy(testBytes, scribbleLocation, sizeof(testBytes));
        });
        SKIP: {
            skip("memory might be unmapped leading to a crash", 1, false);
            ok(*scribbleLocation == 0, "Was erased");
        }
    }    
}


int
su_08_secbuffer(int argc, char *const *argv) {
    plan_tests(kTestCount);
    
    tests();
    
    return 0;
}
