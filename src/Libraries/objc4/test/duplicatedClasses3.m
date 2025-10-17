/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 19, 2022.
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
// TEST_ENV OBJC_DEBUG_DUPLICATE_CLASSES=FATAL
// TEST_CRASHES
/*
TEST_RUN_OUTPUT
objc\[\d+\]: Class DuplicatedClass is implemented in both .+ \(0x[0-9a-f]+\) and .+ \(0x[0-9a-f]+\)\. This may cause spurious casting failures and mysterious crashes\. One of the duplicates must be removed or renamed\.
objc\[\d+\]: HALTED
END
 */

#include "duplicatedClasses0.m"

int main()
{
    fail("should have crashed already");
}
