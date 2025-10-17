/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 5, 2023.
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
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(__GLIBC__) || defined(__ANDROID__)
#include <malloc.h>
#endif

#include "test_util.h"
#include "../util.h"

OPTNONE static void leak_memory(void) {
    (void)!malloc(1024 * 1024 * 1024);
    (void)!malloc(16);
    (void)!malloc(32);
    (void)!malloc(4096);
}

static void *do_work(UNUSED void *p) {
    leak_memory();
    return NULL;
}

int main(void) {
    pthread_t thread[4];
    for (int i = 0; i < 4; i++) {
        pthread_create(&thread[i], NULL, do_work, NULL);
    }
    for (int i = 0; i < 4; i++) {
        pthread_join(thread[i], NULL);
    }

#if defined(__GLIBC__) || defined(__ANDROID__)
    malloc_info(0, stdout);
#endif
}
