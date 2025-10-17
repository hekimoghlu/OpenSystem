/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 19, 2024.
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
// TEST_CONFIG rdar://6416474
// was  rdar://5847976
// was  rdar://6348320 

#include <stdio.h>
#include <Block.h>
#include "test.h"

int main() {
    __block void (^recursive_local_block)(int);
    
    testprintf("recursive_local_block is a local recursive block\n");
    recursive_local_block = ^(int i) {
        testprintf("%d\n", i);
        if (i > 0) {
            recursive_local_block(i - 1);
        }
    };
    
    testprintf("recursive_local_block's address is %p, running it:\n", (void*)recursive_local_block);
    recursive_local_block(5);
    
    testprintf("Creating other_local_block: a local block that calls recursive_local_block\n");
    
    void (^other_local_block)(int) = ^(int i) {
        testprintf("other_local_block running\n");
        recursive_local_block(i);
    };
    
    testprintf("other_local_block's address is %p, running it:\n", (void*)other_local_block);
    
    other_local_block(5);
    
    testprintf("Creating other_copied_block: a Block_copy of a block that will call recursive_local_block\n");
    
    void (^other_copied_block)(int) = Block_copy(^(int i) {
        testprintf("other_copied_block running\n");
        recursive_local_block(i);
    });
    
    testprintf("other_copied_block's address is %p, running it:\n", (void*)other_copied_block);
    
    other_copied_block(5);
    
    __block void (^recursive_copy_block)(int);
    
    testprintf("Creating recursive_copy_block: a Block_copy of a block that will call recursive_copy_block recursively\n");
    
    recursive_copy_block = Block_copy(^(int i) {
        testprintf("%d\n", i);
        if (i > 0) {
            recursive_copy_block(i - 1);
        }
    });
    
    testprintf("recursive_copy_block's address is %p, running it:\n", (void*)recursive_copy_block);
    
    recursive_copy_block(5);
    
    succeed(__FILE__);
}
