/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 23, 2022.
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
//  -*- mode:C; c-basic-offset:4; tab-width:4; intent-tabs-mode:nil;  -*-
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
    __block BobTheStruct fiddly;
    BobTheStruct copy;

    void (^incrementFiddly)() = ^{
        int i;
        for(i=0; i<30; i++) {
            fiddly.ps[i]++;
            fiddly.qs[i]++;
        }
    };
    
    memset(&fiddly, 0xA5, sizeof(fiddly));
    memset(&copy, 0x2A, sizeof(copy));    
    
    int i;
    for(i=0; i<30; i++) {
        fiddly.ps[i] = i * i * i;
        fiddly.qs[i] = -i * i * i;
    }
    
    copy = fiddly;
    incrementFiddly();

    if ( &copy == &fiddly ) {
        fail("struct wasn't copied");
    }
    for(i=0; i<30; i++) {
        //printf("[%d]: fiddly.ps: %lu, copy.ps: %lu, fiddly.qs: %d, copy.qs: %d\n", i, fiddly.ps[i], copy.ps[i], fiddly.qs[i], copy.qs[i]);
        if ( (fiddly.ps[i] != copy.ps[i] + 1) || (fiddly.qs[i] != copy.qs[i] + 1) ) {
            fail("struct contents were not incremented");
        }
    }
    
    succeed(__FILE__);
}
