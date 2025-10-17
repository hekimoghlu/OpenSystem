/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 22, 2024.
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

// TEST_CONFIG MEM=mrc OS=!exclavekit

#include "test.h"
#include <objc/NSObject.h>

static semaphore_t go1;
static semaphore_t go2;
static semaphore_t done;

#define VARCOUNT 100000
static id obj;
static id vars[VARCOUNT];


void *destroyer(void *arg __unused)
{
    while (1) {
        semaphore_wait(go1);
        for (int i = 0; i < VARCOUNT; i++) {
            objc_destroyWeak(&vars[i]);
        }
        semaphore_signal(done);
    }
}


void *deallocator(void *arg __unused)
{
    while (1) {
        semaphore_wait(go2);
        [obj release];
        semaphore_signal(done);
    }
}


void cycle(void)
{
    // rdar://12896779 objc_destroyWeak() versus weak clear in dealloc

    // Clean up from previous cycle - objc_destroyWeak() doesn't set var to nil
    for (int i = 0; i < VARCOUNT; i++) {
        vars[i] = nil;
    }

    obj = [NSObject new];
    for (int i = 0; i < VARCOUNT; i++) {
        objc_storeWeak(&vars[i], obj);
    }

    // let destroyer start before deallocator runs
    semaphore_signal(go1);
    sched_yield();
    semaphore_signal(go2);
    
    semaphore_wait(done);
    semaphore_wait(done);
}


int main()
{
    semaphore_create(mach_task_self(), &go1, 0, 0);
    semaphore_create(mach_task_self(), &go2, 0, 0);
    semaphore_create(mach_task_self(), &done, 0, 0);

    pthread_t th[2];
    pthread_create(&th[1], NULL, deallocator, NULL);
    pthread_create(&th[1], NULL, destroyer, NULL);

    for (int i = 0; i < 100; i++) {
        cycle();
    }

    succeed(__FILE__);
}
