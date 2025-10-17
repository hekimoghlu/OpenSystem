/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 8, 2024.
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

// TEST_CONFIG MEM=mrc
// TEST_CRASHES

// Test badPoolCompat also uses this file.

/*
TEST_RUN_OUTPUT
objc\[\d+\]: [Ii]nvalid or prematurely-freed autorelease pool 0x[0-9a-fA-F]+\. Set a breakpoint .*
objc\[\d+\]: Invalid autorelease pools are a fatal error
objc\[\d+\]: HALTED
END
*/

#include "test.h"

int main()
{
    void *outer = objc_autoreleasePoolPush();
    void *inner = objc_autoreleasePoolPush();
    objc_autoreleasePoolPop(outer);
    objc_autoreleasePoolPop(inner);

#if !OLD
    fail("should have crashed already with new SDK");
#else
    // should only warn once
    outer = objc_autoreleasePoolPush();
    inner = objc_autoreleasePoolPush();
    objc_autoreleasePoolPop(outer);
    objc_autoreleasePoolPop(inner);

    succeed(__FILE__);
#endif
}

