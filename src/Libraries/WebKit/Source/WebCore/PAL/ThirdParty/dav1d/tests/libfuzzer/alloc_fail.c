/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 18, 2023.
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
#include "config.h"

#include <stddef.h>
#include <stdlib.h>
#include <errno.h>
#include <pthread.h>

#include "alloc_fail.h"

static int fail_probability;

void dav1d_setup_alloc_fail(unsigned seed, unsigned probability) {
    srand(seed);

    while (probability >= RAND_MAX)
        probability >>= 1;

    fail_probability = probability;
}

void * __wrap_malloc(size_t);

void * __wrap_malloc(size_t sz) {
    if (rand() < fail_probability)
        return NULL;
    return malloc(sz);
}

#if defined(HAVE_POSIX_MEMALIGN)
int __wrap_posix_memalign(void **memptr, size_t alignment, size_t size);

int __wrap_posix_memalign(void **memptr, size_t alignment, size_t size) {
    if (rand() < fail_probability)
        return ENOMEM;
    return posix_memalign(memptr, alignment, size);
}
#else
#error "HAVE_POSIX_MEMALIGN required"
#endif

int __wrap_pthread_create(pthread_t *, const pthread_attr_t *,
                          void *(*) (void *), void *);

int __wrap_pthread_create(pthread_t *thread, const pthread_attr_t *attr,
                          void *(*start_routine) (void *), void *arg)
{
    if (rand() < (fail_probability + RAND_MAX/16))
        return EAGAIN;

    return pthread_create(thread, attr, start_routine, arg);
}

int __wrap_pthread_mutex_init(pthread_mutex_t *,
                              const pthread_mutexattr_t *);

int __wrap_pthread_mutex_init(pthread_mutex_t *restrict mutex,
                              const pthread_mutexattr_t *restrict attr)
{
    if (rand() < (fail_probability + RAND_MAX/8))
        return ENOMEM;

    return pthread_mutex_init(mutex, attr);
}

int __wrap_pthread_cond_init(pthread_cond_t *,
                             const pthread_condattr_t *);

int __wrap_pthread_cond_init(pthread_cond_t *restrict cond,
                             const pthread_condattr_t *restrict attr)
{
    if (rand() < (fail_probability + RAND_MAX/16))
        return ENOMEM;

    return pthread_cond_init(cond, attr);
}
