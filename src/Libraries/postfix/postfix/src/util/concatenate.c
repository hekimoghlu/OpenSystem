/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 14, 2024.
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
/* System library. */

#include <sys_defs.h>
#include <stdlib.h>			/* 44BSD stdarg.h uses abort() */
#include <stdarg.h>
#include <string.h>

/* Utility library. */

#include "mymalloc.h"
#include "stringops.h"
#include "compat_va_copy.h"

/* concatenate - concatenate null-terminated list of strings */

char   *concatenate(const char *arg0,...)
{
    char   *result;
    va_list ap;
    va_list ap2;
    ssize_t len;
    char   *arg;

    /*
     * Initialize argument lists.
     */
    va_start(ap, arg0);
    VA_COPY(ap2, ap);

    /*
     * Compute the length of the resulting string.
     */
    len = strlen(arg0);
    while ((arg = va_arg(ap, char *)) != 0)
	len += strlen(arg);
    va_end(ap);

    /*
     * Build the resulting string. Don't care about wasting a CPU cycle.
     */
    result = mymalloc(len + 1);
    strcpy(result, arg0);
    while ((arg = va_arg(ap2, char *)) != 0)
	strcat(result, arg);
    va_end(ap2);
    return (result);
}
