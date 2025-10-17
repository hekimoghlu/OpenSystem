/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 21, 2025.
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
#include <config.h>

#include "roken.h"

ROKEN_LIB_FUNCTION int ROKEN_LIB_CALL
roken_concat (char *s, size_t len, ...)
{
    int ret;
    va_list args;

    va_start(args, len);
    ret = roken_vconcat (s, len, args);
    va_end(args);
    return ret;
}

ROKEN_LIB_FUNCTION int ROKEN_LIB_CALL
roken_vconcat (char *s, size_t len, va_list args)
{
    const char *a;

    while ((a = va_arg(args, const char*))) {
	size_t n = strlen (a);

	if (n >= len)
	    return -1;
	memcpy (s, a, n);
	s += n;
	len -= n;
    }
    *s = '\0';
    return 0;
}

ROKEN_LIB_FUNCTION size_t ROKEN_LIB_CALL
roken_vmconcat (char **s, size_t max_len, va_list args)
{
    const char *a;
    char *p, *q;
    size_t len = 0;
    *s = NULL;
    p = malloc(1);
    if(p == NULL)
	return 0;
    len = 1;
    while ((a = va_arg(args, const char*))) {
	size_t n = strlen (a);

	if(max_len && len + n > max_len){
	    free(p);
	    return 0;
	}
	q = realloc(p, len + n);
	if(q == NULL){
	    free(p);
	    return 0;
	}
	p = q;
	memcpy (p + len - 1, a, n);
	len += n;
    }
    p[len - 1] = '\0';
    *s = p;
    return len;
}

ROKEN_LIB_FUNCTION size_t ROKEN_LIB_CALL
roken_mconcat (char **s, size_t max_len, ...)
{
    size_t ret;
    va_list args;

    va_start(args, max_len);
    ret = roken_vmconcat (s, max_len, args);
    va_end(args);
    return ret;
}
