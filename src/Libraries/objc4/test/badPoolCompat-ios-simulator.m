/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 4, 2023.
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

// Run test badPool as if it were built with an old SDK.

// TEST_CONFIG MEM=mrc OS=iphonesimulator ARCH=x86_64
// TEST_CRASHES
// TEST_CFLAGS -DOLD=1 -Xlinker -platform_version -Xlinker ios-simulator -Xlinker 9.0 -Xlinker 9.0 -mios-simulator-version-min=9.0

/*
TEST_RUN_OUTPUT
objc\[\d+\]: Invalid or prematurely-freed autorelease pool 0x[0-9a-fA-f]+\. Set a breakpoint .*
objc\[\d+\]: Proceeding anyway.*
OK: badPool.m
END
*/

#include "badPool.m"
