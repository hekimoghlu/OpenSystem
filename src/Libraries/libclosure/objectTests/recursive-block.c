/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 17, 2023.
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

#include <stdio.h>
#include <Block.h>
#include <Block_private.h>
#include <stdlib.h>
#include "test.h"

int cumulation = 0;

int doSomething(int i) {
    cumulation += i;
    return cumulation;
}

void dirtyStack() {
    int i = (int)random();
    int j = doSomething(i);
    int k = doSomething(j);
    doSomething(i + j + k);
}

typedef void (^voidVoid)(void);

voidVoid testFunction() {
    int i = (int)random();
    __block voidVoid inner = ^{ doSomething(i); };
    //printf("inner, on stack, is %p\n", (void*)inner);
    /*__block*/ voidVoid outer = ^{
        //printf("will call inner block %p\n", (void *)inner);
        inner();
    };
    //printf("outer looks like: %s\n", _Block_dump(outer));
    voidVoid result = Block_copy(outer);
    //Block_release(inner);
    return result;
}


int main() {
    voidVoid block = testFunction();
    dirtyStack();
    block();
    Block_release(block);

    succeed(__FILE__);
}
