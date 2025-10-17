/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 18, 2024.
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

#include <stdio.h>
#include <assert.h>
#include "thread-test.h"

#define NTHREADS 10


extern int add1(int, int);

static sem_t done;


static void *start_routine(void *arg)
{
    int x, status;
    x = add1(40, 2);
    assert(x == 42);

    status = sem_post(&done);
    assert(status == 0);

    return arg;
}

int main(void)
{
    pthread_t th;
    int i, status = sem_init(&done, 0, 0);
    assert(status == 0);

    printf("starting\n");
    fflush(stdout);
    for (i = 0; i < NTHREADS; i++) {
        status = pthread_create(&th, NULL, start_routine, NULL);
        assert(status == 0);
    }
    for (i = 0; i < NTHREADS; i++) {
        status = sem_wait(&done);
        assert(status == 0);
    }
    printf("done\n");
    return 0;
}
