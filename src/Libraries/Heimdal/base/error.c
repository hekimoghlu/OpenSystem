/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 19, 2022.
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
#include "baselocl.h"

struct heim_error {
    int error_code;
    heim_string_t msg;
    struct heim_error *next;
};

static void
error_dealloc(void *ptr)
{
    struct heim_error *p = ptr;
    heim_release(p->msg);
    heim_release(p->next);
}

static int
error_cmp(void *a, void *b)
{
    struct heim_error *ap = a, *bp = b;
    if (ap->error_code == bp->error_code)
	return 0;
    return heim_cmp(ap->msg, bp->msg);
}

static unsigned long
error_hash(void *ptr)
{
    struct heim_error *p = ptr;
    return p->error_code;
}

struct heim_type_data _heim_error_object = {
    HEIM_TID_ERROR,
    "error-object",
    NULL,
    error_dealloc,
    NULL,
    error_cmp,
    error_hash
};

heim_error_t
heim_error_create(int error_code, const char *fmt, ...)
{
    heim_error_t e;
    va_list ap;

    va_start(ap, fmt);
    e = heim_error_createv(error_code, fmt, ap);
    va_end(ap);

    return e;
}

heim_error_t
heim_error_createv(int error_code, const char *fmt, va_list ap)
{
    heim_error_t e;
    char *str;
    int len;

    str = malloc(1024);
    if (str == NULL)
        return NULL;
    len = vsnprintf(str, 1024, fmt, ap);
    if (len < 0) {
        free(str);
	return NULL;
    }

    e = _heim_alloc_object(&_heim_error_object, sizeof(struct heim_error));
    if (e) {
	e->msg = heim_string_create(str);
	e->error_code = error_code;
    }
    free(str);

    return e;
}

heim_string_t
heim_error_copy_string(heim_error_t error)
{
    /* XXX concat all strings */
    return heim_retain(error->msg);
}

int
heim_error_get_code(heim_error_t error)
{
    return error->error_code;
}

heim_error_t
heim_error_append(heim_error_t top, heim_error_t append)
{
    if (top->next)
	heim_release(top->next);
    top->next = heim_retain(append);
    return top;
}
