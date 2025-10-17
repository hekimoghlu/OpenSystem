/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 22, 2024.
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
#include <stdlib.h>

#include "abts.h"
#include "testutil.h"
#include "apr_pools.h"

apr_pool_t *p;

void apr_assert_success(abts_case* tc, const char* context, apr_status_t rv, 
                        int lineno) 
{
    if (rv == APR_ENOTIMPL) {
        abts_not_impl(tc, context, lineno);
    } else if (rv != APR_SUCCESS) {
        char buf[STRING_MAX], ebuf[128];
        sprintf(buf, "%s (%d): %s\n", context, rv,
                apr_strerror(rv, ebuf, sizeof ebuf));
        abts_fail(tc, buf, lineno);
    }
}

void initialize(void) {
    apr_initialize();
    atexit(apr_terminate);
    
    apr_pool_create(&p, NULL);
}
