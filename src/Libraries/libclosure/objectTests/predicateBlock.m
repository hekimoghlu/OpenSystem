/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 25, 2021.
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
// TEST_CFLAGS -framework Foundation

#import <Foundation/Foundation.h>
#import <Block_private.h>
#import "test.h"

typedef void (^void_block_t)(void);

int main () {
    void_block_t c = ^{ NSLog(@"Hello!"); };
    
    //printf("global block c looks like: %s\n", _Block_dump(c));
    int j;
    for (j = 0; j < 1000; j++)
    {
        void_block_t d = [c copy];
        //if (j == 0) printf("copy looks like %s\n", _Block_dump(d));
        [d release];
    }

    succeed(__FILE__);
}
