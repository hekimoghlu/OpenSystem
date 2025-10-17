/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 7, 2022.
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

#include <stdarg.h>
#include <stdlib.h>
#include "roken.h"

struct rk_strpool {
    char *str;
    size_t len;
};

/*
 *
 */

ROKEN_LIB_FUNCTION void ROKEN_LIB_CALL
rk_strpoolfree(struct rk_strpool *p)
{
    if (p->str) {
	free(p->str);
	p->str = NULL;
    }
    free(p);
}

/*
 *
 */

ROKEN_LIB_FUNCTION struct rk_strpool * ROKEN_LIB_CALL
rk_strpoolprintf(struct rk_strpool *p, const char *fmt, ...)
{
    va_list ap;
    char *str, *str2;
    int len;

    if (p == NULL) {
	p = malloc(sizeof(*p));
	if (p == NULL)
	    return NULL;
	p->str = NULL;
	p->len = 0;
    }
    va_start(ap, fmt);
    len = vasprintf(&str, fmt, ap);
    va_end(ap);
    if (str == NULL) {
	rk_strpoolfree(p);
	return NULL;
    }
    str2 = realloc(p->str, len + p->len + 1);
    if (str2 == NULL) {
	rk_strpoolfree(p);
	return NULL;
    }
    p->str = str2;
    memcpy(p->str + p->len, str, len + 1);
    p->len += len;
    free(str);
    return p;
}

/*
 *
 */

ROKEN_LIB_FUNCTION char * ROKEN_LIB_CALL
rk_strpoolcollect(struct rk_strpool *p)
{
    char *str;
    if (p == NULL)
	return strdup("");
    str = p->str;
    p->str = NULL;
    free(p);
    return str;
}
