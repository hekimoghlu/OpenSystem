/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 14, 2024.
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
// TEST_CONFIG rdar://6405500

#import <stdio.h>
#import <stdlib.h>
#import <dispatch/dispatch.h>
#import "test.h"

int main () {
    __block void (^blockFu)(size_t t);
    blockFu = ^(size_t t){
        if (t == 20) {
            succeed(__FILE__);
        } else {
            dispatch_async(dispatch_get_main_queue(), ^{ blockFu(20); });
        }
    };
    
    dispatch_apply(10, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), blockFu);
    dispatch_main();
    fail("unreachable");
}
