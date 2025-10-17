/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 10, 2022.
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

// TEST_CONFIG OS=!exclavekit

#include "test.h"

#include <objc/NSObject.h>
#include <mach/mach.h>
#include <pthread.h>
#include <sys/time.h>
#include <objc/runtime.h>
#include <objc/objc-sync.h>

// Basic @synchronized tests.


#define WAIT_SEC 3

static id obj;
static semaphore_t go;
static semaphore_t stop;

void *thread(void *arg __unused)
{
    int err;
    BOOL locked;

    // non-blocking sync_enter
    err = objc_sync_enter(obj);
    testassert(err == OBJC_SYNC_SUCCESS);

    // recursive try_sync_enter
    locked = objc_sync_try_enter(obj);
    testassert(locked);
    err = objc_sync_exit(obj);
    testassert(err == OBJC_SYNC_SUCCESS);

    semaphore_signal(go);
    // main thread: sync_exit of object locked on some other thread
    semaphore_wait(stop);
    
    err = objc_sync_exit(obj);
    testassert(err == OBJC_SYNC_SUCCESS);
    err = objc_sync_enter(obj);
    testassert(err == OBJC_SYNC_SUCCESS);

    semaphore_signal(go);
    // main thread: blocking sync_enter 
    testassert(WAIT_SEC/3*3 == WAIT_SEC);
    sleep(WAIT_SEC/3);
    // recursive enter while someone waits
    err = objc_sync_enter(obj);
    testassert(err == OBJC_SYNC_SUCCESS);
    sleep(WAIT_SEC/3);
    // recursive exit while someone waits
    err = objc_sync_exit(obj);
    testassert(err == OBJC_SYNC_SUCCESS);
    sleep(WAIT_SEC/3);
    // sync_exit while someone waits
    err = objc_sync_exit(obj);
    testassert(err == OBJC_SYNC_SUCCESS);
    
    return NULL;
}

int main()
{
    pthread_t th;
    int err;
    struct timeval start, end;
    BOOL locked;

    obj = [[NSObject alloc] init];

    // sync_exit of never-locked object
    err = objc_sync_exit(obj);
    testassert(err == OBJC_SYNC_NOT_OWNING_THREAD_ERROR);

    semaphore_create(mach_task_self(), &go, 0, 0);
    semaphore_create(mach_task_self(), &stop, 0, 0);
    pthread_create(&th, NULL, &thread, NULL);
    semaphore_wait(go);

    // sync_exit of object locked on some other thread
    err = objc_sync_exit(obj);
    testassert(err == OBJC_SYNC_NOT_OWNING_THREAD_ERROR);

    semaphore_signal(stop);
    semaphore_wait(go);

    // contended try_sync_enter
    locked = objc_sync_try_enter(obj);
    testassert(!locked);

    // blocking sync_enter
    gettimeofday(&start, NULL);
    err = objc_sync_enter(obj);
    gettimeofday(&end, NULL);
    testassert(err == OBJC_SYNC_SUCCESS);
    // should have waited more than WAIT_SEC but less than WAIT_SEC+1
    // fixme hack: sleep(1) is ending 500 usec too early on x86_64 buildbot
    // (rdar://6456975)
    testassert(end.tv_sec*1000000LL+end.tv_usec >= 
               start.tv_sec*1000000LL+start.tv_usec + WAIT_SEC*1000000LL
               - 3*500 /*hack*/);
    testassert(end.tv_sec*1000000LL+end.tv_usec < 
               start.tv_sec*1000000LL+start.tv_usec + (1+WAIT_SEC)*1000000LL);

    err = objc_sync_exit(obj);
    testassert(err == OBJC_SYNC_SUCCESS);

    err = objc_sync_exit(obj);
    testassert(err == OBJC_SYNC_NOT_OWNING_THREAD_ERROR);

    succeed(__FILE__);
}
