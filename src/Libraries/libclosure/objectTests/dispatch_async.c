/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 17, 2022.
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
// rdar://problem/6371811

#include <CoreFoundation/CoreFoundation.h>
#include <dispatch/dispatch.h>
#include <unistd.h>
#include <Block.h>
#include "test.h"

void EnqueueStuff(dispatch_queue_t q)
{
    __block CFIndex counter;
    
    // above call has a side effect: it works around:
    // <rdar://problem/6225809> __block variables not implicitly imported into intermediate scopes
    dispatch_async(q, ^{
        counter = 0;
    });
    
    
    dispatch_async(q, ^{
        //printf("outer block.\n");
        counter++;
        dispatch_async(q, ^{
            //printf("inner block.\n");
            counter--;
            if(counter == 0) {
                succeed(__FILE__);
            }
        });
        if(counter == 0) {
            fail("already done? inconceivable!");
        }
    });        
}

int main () {
    dispatch_queue_t q = dispatch_queue_create("queue", NULL);

    EnqueueStuff(q);
    
    dispatch_main();
    fail("unreachable");
}
