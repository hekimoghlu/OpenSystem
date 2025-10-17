/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 4, 2021.
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
#include "apr.h"
#include "apr_file_io.h"
#include "apr.h"

#if APR_HAVE_STDLIB_H
#include <stdlib.h>
#endif

int main(void)
{
    char buf[256];
    apr_file_t *err;
    apr_pool_t *p;

    apr_initialize();
    atexit(apr_terminate);

    apr_pool_create(&p, NULL);
    apr_file_open_stdin(&err, p);

    while (1) {
        apr_size_t length = 256;
        apr_file_read(err, buf, &length);
    }
    exit(0); /* just to keep the compiler happy */
}
