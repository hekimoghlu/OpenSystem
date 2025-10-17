/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 11, 2024.
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
 *  byrefcopyint.c
 *  testObjects
 *
 *  Created by Blaine Garst on 12/1/08.
 *  Copyright 2008 __MyCompanyName__. All rights reserved.
 *
 */

//
//  byrefcopyid.m
//  testObjects
//
//  Created by Blaine Garst on 5/13/08.
//  Copyright 2008 __MyCompanyName__. All rights reserved.
//

// TEST_CONFIG

// Tests copying of blocks with byref ints
// rdar://6414583

#include <stdio.h>
#include <string.h>
#include <Block.h>
#include <Block_private.h>
#include "test.h"

typedef void (^voidVoid)(void);

voidVoid dummy;

void callVoidVoid(voidVoid closure) {
    closure();
}


voidVoid testRoutine(const char *whoami) {
    __block size_t dumbo = strlen(whoami);
    dummy = ^{
        //printf("incring dumbo from %d\n", dumbo);
        ++dumbo;
    };
    
    
    voidVoid copy = Block_copy(dummy);
    

    return copy;
}

int main(int argc __unused, char *argv[]) {
    voidVoid array[100];
    for (int i = 0; i <  100; ++i) {
        array[i] = testRoutine(argv[0]);
        array[i]();
    }
    for (int i = 0; i <  100; ++i) {
        Block_release(array[i]);
    }
    
    succeed(__FILE__);
}
