/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 15, 2022.
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
#include <sys/time.h>
#ifdef PTEST_USE_THREAD
# include <pthread.h>
static pthread_mutex_t mutex1 = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t cond1 = PTHREAD_COND_INITIALIZER;
static int remaining;
#endif


extern int add1(int, int);


static double time_delta(struct timeval *stop, struct timeval *start)
{
    return (stop->tv_sec - start->tv_sec) +
        1e-6 * (stop->tv_usec - start->tv_usec);
}

static double measure(void)
{
    long long i, iterations;
    int result;
    struct timeval start, stop;
    double elapsed;

    add1(0, 0);   /* prepare off-line */

    i = 0;
    iterations = 1000;
    result = gettimeofday(&start, NULL);
    assert(result == 0);

    while (1) {
        for (; i < iterations; i++) {
            add1(((int)i) & 0xaaaaaa, ((int)i) & 0x555555);
        }
        result = gettimeofday(&stop, NULL);
        assert(result == 0);

        elapsed = time_delta(&stop, &start);
        assert(elapsed >= 0.0);
        if (elapsed > 2.5)
            break;
        iterations = iterations * 3 / 2;
    }

    return elapsed / (double)iterations;
}

static void *start_routine(void *arg)
{
    double t = measure();
    printf("time per call: %.3g\n", t);

#ifdef PTEST_USE_THREAD
    pthread_mutex_lock(&mutex1);
    remaining -= 1;
    if (!remaining)
        pthread_cond_signal(&cond1);
    pthread_mutex_unlock(&mutex1);
#endif

    return arg;
}


int main(void)
{
#ifndef PTEST_USE_THREAD
    start_routine(0);
#else
    pthread_t th;
    int i, status;

    add1(0, 0);   /* this is the main thread */

    remaining = PTEST_USE_THREAD;
    for (i = 0; i < PTEST_USE_THREAD; i++) {
        status = pthread_create(&th, NULL, start_routine, NULL);
        assert(status == 0);
    }
    pthread_mutex_lock(&mutex1);
    while (remaining)
        pthread_cond_wait(&cond1, &mutex1);
    pthread_mutex_unlock(&mutex1);
#endif
    return 0;
}
