/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 14, 2022.
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

extern int add2(int, int, int);
extern int add3(int, int, int, int);

static sem_t done;


static void *start_routine_2(void *arg)
{
    int x, status;
    x = add2(40, 2, 100);
    assert(x == 142);

    status = sem_post(&done);
    assert(status == 0);

    return arg;
}

static void *start_routine_3(void *arg)
{
    int x, status;
    x = add3(1000, 200, 30, 4);
    assert(x == 1234);

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
    for (i = 0; i < 10; i++) {
        status = pthread_create(&th, NULL, start_routine_2, NULL);
        assert(status == 0);
        status = pthread_create(&th, NULL, start_routine_3, NULL);
        assert(status == 0);
    }
    for (i = 0; i < 20; i++) {
        status = sem_wait(&done);
        assert(status == 0);
    }
    printf("done\n");
    fflush(stdout);   /* this is occasionally needed on Windows */
    return 0;
}
