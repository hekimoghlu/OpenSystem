/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 19, 2024.
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
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "atf-c/error.h"

#include "detail/sanity.h"

/* Theoretically, there can only be a single error intance at any given
 * point in time, because errors are raised at one point and must be
 * handled immediately.  If another error has to be raised during the
 * handling process, something else has to be done with the previous
 * error.
 *
 * This is per-thread information and will break threaded tests, but we
 * currently do not have any threading support; therefore, this is fine. */
static bool error_on_flight = false;

/* ---------------------------------------------------------------------
 * Auxiliary functions.
 * --------------------------------------------------------------------- */

static
void
error_format(const atf_error_t err, char *buf, size_t buflen)
{
    PRE(err != NULL);
    snprintf(buf, buflen, "Error '%s'", err->m_type);
}

static
bool
error_init(atf_error_t err, const char *type, void *data, size_t datalen,
           void (*format)(const atf_error_t, char *, size_t))
{
    bool ok;

    PRE(data != NULL || datalen == 0);
    PRE(datalen != 0 || data == NULL);

    err->m_free = false;
    err->m_type = type;
    err->m_format = (format == NULL) ? error_format : format;

    ok = true;
    if (data == NULL) {
        err->m_data = NULL;
    } else {
        err->m_data = malloc(datalen);
        if (err->m_data == NULL) {
            ok = false;
        } else
            memcpy(err->m_data, data, datalen);
    }

    return ok;
}

/* ---------------------------------------------------------------------
 * The "atf_error" type.
 * --------------------------------------------------------------------- */

atf_error_t
atf_error_new(const char *type, void *data, size_t datalen,
              void (*format)(const atf_error_t, char *, size_t))
{
    atf_error_t err;

    PRE(!error_on_flight);
    PRE(data != NULL || datalen == 0);
    PRE(datalen != 0 || data == NULL);

    err = malloc(sizeof(*err));
    if (err == NULL)
        err = atf_no_memory_error();
    else {
        if (!error_init(err, type, data, datalen, format)) {
            free(err);
            err = atf_no_memory_error();
        } else {
            err->m_free = true;
            error_on_flight = true;
        }
    }

    INV(err != NULL);
    POST(error_on_flight);
    return err;
}

void
atf_error_free(atf_error_t err)
{
    bool freeit;

    PRE(error_on_flight);
    PRE(err != NULL);

    freeit = err->m_free;

    if (err->m_data != NULL)
        free(err->m_data);

    if (freeit)
        free(err);

    error_on_flight = false;
}

atf_error_t
atf_no_error(void)
{
    return NULL;
}

bool
atf_is_error(const atf_error_t err)
{
    return err != NULL;
}

bool
atf_error_is(const atf_error_t err, const char *type)
{
    PRE(err != NULL);

    return strcmp(err->m_type, type) == 0;
}

const void *
atf_error_data(const atf_error_t err)
{
    PRE(err != NULL);

    return err->m_data;
}

void
atf_error_format(const atf_error_t err, char *buf, size_t buflen)
{
    PRE(err != NULL);
    err->m_format(err, buf, buflen);
}

/* ---------------------------------------------------------------------
 * Common error types.
 * --------------------------------------------------------------------- */

/*
 * The "libc" error.
 */

struct atf_libc_error_data {
    int m_errno;
    char m_what[4096];
};
typedef struct atf_libc_error_data atf_libc_error_data_t;

static
void
libc_format(const atf_error_t err, char *buf, size_t buflen)
{
    const atf_libc_error_data_t *data;

    PRE(atf_error_is(err, "libc"));

    data = atf_error_data(err);
    snprintf(buf, buflen, "%s: %s", data->m_what, strerror(data->m_errno));
}

atf_error_t
atf_libc_error(int syserrno, const char *fmt, ...)
{
    atf_error_t err;
    atf_libc_error_data_t data;
    va_list ap;

    data.m_errno = syserrno;
    va_start(ap, fmt);
    vsnprintf(data.m_what, sizeof(data.m_what), fmt, ap);
    va_end(ap);

    err = atf_error_new("libc", &data, sizeof(data), libc_format);

    return err;
}

int
atf_libc_error_code(const atf_error_t err)
{
    const struct atf_libc_error_data *data;

    PRE(atf_error_is(err, "libc"));

    data = atf_error_data(err);

    return data->m_errno;
}

const char *
atf_libc_error_msg(const atf_error_t err)
{
    const struct atf_libc_error_data *data;

    PRE(atf_error_is(err, "libc"));

    data = atf_error_data(err);

    return data->m_what;
}

/*
 * The "no_memory" error.
 */

static struct atf_error no_memory_error;

static
void
no_memory_format(const atf_error_t err, char *buf, size_t buflen)
{
    PRE(atf_error_is(err, "no_memory"));

    snprintf(buf, buflen, "Not enough memory");
}

atf_error_t
atf_no_memory_error(void)
{
    PRE(!error_on_flight);

    error_init(&no_memory_error, "no_memory", NULL, 0,
               no_memory_format);

    error_on_flight = true;
    return &no_memory_error;
}
