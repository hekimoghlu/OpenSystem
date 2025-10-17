/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 16, 2022.
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
/*
 *  copynull.c
 *  testObjects
 *
 *  Created by Blaine Garst on 10/15/08.
 *  Copyright 2008 __MyCompanyName__. All rights reserved.
 *
 */

// TEST_CONFIG

// rdar://6295848
 
#import <stdio.h>
#import <Block.h>
#import <Block_private.h> 
#import "test.h"

int main() {
    
    void (^block)(void) = (void (^)(void))0;
    void (^blockcopy)(void) = Block_copy(block);
    
    if (blockcopy != (void (^)(void))0) {
        fail("whoops, somehow we copied NULL!");
    }
    // make sure we can also
    Block_release(blockcopy);
    // and more secretly
    //_Block_destroy(blockcopy);

    succeed(__FILE__);
}
