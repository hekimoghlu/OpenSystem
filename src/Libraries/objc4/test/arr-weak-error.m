/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 8, 2023.
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
// TEST_ENV OBJC_DEBUG_WEAK_ERRORS=fatal
// TEST_CRASHES
/*
TEST_RUN_OUTPUT
objc\[\d+\]: __weak variable at 0x[0-9a-f]+ holds 0x[0-9a-f]+ instead of 0x[0-9a-f]+. This is probably incorrect use of objc_storeWeak\(\) and objc_loadWeak\(\).
objc\[\d+\]: HALTED
END
*/

#include "test.h"

#include <objc/NSObject.h>

int main()
{
    id weakVar = nil;
    @autoreleasepool {
        id obj = [NSObject new];
        objc_storeWeak(&weakVar, obj);
        weakVar = [NSObject new];
        [obj release];
    }

    fail("should have crashed");
}

