/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 1, 2022.
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
#include <stdlib.h>

static void *thread_func(void *arg) {
    arg = realloc(arg, 1024);
    if (!arg) {
        exit(EXIT_FAILURE);
    }

    free(arg);

    return NULL;
}

int main(void) {
    void *mem = realloc(NULL, 12);
    if (!mem) {
        return EXIT_FAILURE;
    }

    pthread_t thread;
    int r = pthread_create(&thread, NULL, thread_func, mem);
    if (r != 0) {
        return EXIT_FAILURE;
    }

    r = pthread_join(thread, NULL);
    if (r != 0) {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
