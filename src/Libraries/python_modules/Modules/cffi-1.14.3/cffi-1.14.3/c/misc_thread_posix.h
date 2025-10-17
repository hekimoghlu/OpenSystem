/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 7, 2025.
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
#include "misc_thread_common.h"


static pthread_key_t cffi_tls_key;

static void init_cffi_tls(void)
{
    if (pthread_key_create(&cffi_tls_key, &cffi_thread_shutdown) != 0)
        PyErr_SetString(PyExc_OSError, "pthread_key_create() failed");
}

static struct cffi_tls_s *_make_cffi_tls(void)
{
    void *p = calloc(1, sizeof(struct cffi_tls_s));
    if (p == NULL)
        return NULL;
    if (pthread_setspecific(cffi_tls_key, p) != 0) {
        free(p);
        return NULL;
    }
    return p;
}

static struct cffi_tls_s *get_cffi_tls(void)
{
    void *p = pthread_getspecific(cffi_tls_key);
    if (p == NULL)
        p = _make_cffi_tls();
    return (struct cffi_tls_s *)p;
}

#define save_errno      save_errno_only
#define restore_errno   restore_errno_only
