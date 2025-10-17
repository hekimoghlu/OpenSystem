/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 17, 2025.
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
// TEST_CONFIG

#import <stdio.h>
#import <Block.h>
#import "test.h"

int global;

void (^gblock)(int) = ^(int x){ global = x; };

int main() {
    gblock(1);
    if (global != 1) {
        fail("did not set global to 1");
    }
    void (^gblockcopy)(int) = Block_copy(gblock);
    if (gblockcopy != gblock) {
        fail("global copy %p not a no-op %p", (void *)gblockcopy, (void *)gblock);
    }
    Block_release(gblockcopy);
    gblock(3);
    if (global != 3) {
        fail("did not set global to 3");
    }
    gblockcopy = Block_copy(gblock);
    gblockcopy(5);
    if (global != 5) {
        fail("did not set global to 5");
    }

    succeed(__FILE__);
}

