/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 5, 2022.
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
#import <stdlib.h>
#import <string.h>
#import "test.h"

typedef struct {
    unsigned long ps[30];
    int qs[30];
} BobTheStruct;

int main () {
    BobTheStruct inny;
    BobTheStruct outty;
    BobTheStruct (^copyStruct)(BobTheStruct);
    int i;
    
    memset(&inny, 0xA5, sizeof(inny));
    memset(&outty, 0x2A, sizeof(outty));    
    
    for(i=0; i<30; i++) {
        inny.ps[i] = i * i * i;
        inny.qs[i] = -i * i * i;
    }
    
    copyStruct = ^(BobTheStruct aBigStruct){ return aBigStruct; };  // pass-by-value intrinsically copies the argument
    
    outty = copyStruct(inny);

    if ( &inny == &outty ) {
        fail("struct wasn't copied");
    }
    for(i=0; i<30; i++) {
        if ( (inny.ps[i] != outty.ps[i]) || (inny.qs[i] != outty.qs[i]) ) {
            fail("struct contents did not match.");
        }
    }
    
    succeed(__FILE__);
}
