/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 17, 2022.
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
//
//  nullblockisa.m
//  testObjects
//
//  Created by Blaine Garst on 9/24/08.
//  Copyright 2008 __MyCompanyName__. All rights reserved.
//
// TEST_CONFIG
// rdar://6244520

#include <stdio.h>
#include <stdlib.h>
#include <Block_private.h>
#include "test.h"

void check(void (^b)(void)) {
    struct _custom {
        struct Block_layout layout;
        struct Block_byref *innerp;
    } *custom  = (struct _custom *)(void *)(b);
    //printf("block is at %p, size is %lx, inner is %p\n", (void *)b, Block_size(b), innerp);
    if (custom->innerp->isa != (void *)NULL) {
        fail("not a NULL __block isa");
    }
    return;
}
        
int main() {

   __block int i;
   
   check(^{ printf("%d\n", ++i); });

   succeed(__FILE__);
}
   
