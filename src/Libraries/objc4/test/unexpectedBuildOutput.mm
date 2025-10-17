/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 14, 2024.
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

// TEST_CONFIG MEM=arc
// TEST_CFLAGS -framework Foundation

#include "test.h"
#include <Foundation/Foundation.h>

int main()
{
    NSString *unexpectedBuildOutputFile = @"../../unexpected-build-output";
    if ([[NSFileManager defaultManager] fileExistsAtPath: unexpectedBuildOutputFile]) {
        NSData *data = [NSData dataWithContentsOfFile: unexpectedBuildOutputFile];
        if (!data)
            data = [@"<unable to read unexpected-build-output>" dataUsingEncoding: NSUTF8StringEncoding];

        [[NSFileHandle fileHandleWithStandardOutput] writeData: data];

        fail(__FILE__);
    } else {
        succeed(__FILE__);
    }
}
