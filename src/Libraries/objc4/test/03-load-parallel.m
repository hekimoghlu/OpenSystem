/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 18, 2022.
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
#include "test.h"

#include <dlfcn.h>
#include <pthread.h>

#ifndef COUNT
#error -DCOUNT=c missing
#endif

extern atomic_int state;

void *thread(void *arg)
{
    uintptr_t num = (uintptr_t)arg;
    char *buf;

    asprintf(&buf, "load-parallel%lu.dylib", (unsigned long)num);
    testprintf("%s\n", buf);
    void *dlh = dlopen(buf, RTLD_LAZY);
    if (!dlh) {
        fail("dlopen failed: %s", dlerror());
    }
    free(buf);

    return NULL;
}

int main()
{
    pthread_t t[COUNT];
    uintptr_t i;

    for (i = 0; i < COUNT; i++) {
        pthread_create(&t[i], NULL, thread, (void *)i);
    }

    for (i = 0; i < COUNT; i++) {
        pthread_join(t[i], NULL);
    }

    testprintf("loaded %d/%d\n", (int)state, COUNT*26);
    testassert(state == COUNT*26);

    succeed(__FILE__);
}
