/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 7, 2022.
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
#include <errno.h>
#include <limits.h>
#include <string.h>
#include <stdlib.h>

#include "atf-c/error.h"

#include "dynstr.h"
#include "sanity.h"
#include "text.h"

atf_error_t
atf_text_for_each_word(const char *instr, const char *sep,
                       atf_error_t (*func)(const char *, void *),
                       void *data)
{
    atf_error_t err;
    char *str, *str2, *last;

    str = strdup(instr);
    if (str == NULL) {
        err = atf_no_memory_error();
        goto out;
    }

    err = atf_no_error();
    str2 = strtok_r(str, sep, &last);
    while (str2 != NULL && !atf_is_error(err)) {
        err = func(str2, data);
        str2 = strtok_r(NULL, sep, &last);
    }

    free(str);
out:
    return err;
}

atf_error_t
atf_text_format(char **dest, const char *fmt, ...)
{
    atf_error_t err;
    va_list ap;

    va_start(ap, fmt);
    err = atf_text_format_ap(dest, fmt, ap);
    va_end(ap);

    return err;
}

atf_error_t
atf_text_format_ap(char **dest, const char *fmt, va_list ap)
{
    atf_error_t err;
    atf_dynstr_t tmp;
    va_list ap2;

    va_copy(ap2, ap);
    err = atf_dynstr_init_ap(&tmp, fmt, ap2);
    va_end(ap2);
    if (!atf_is_error(err))
        *dest = atf_dynstr_fini_disown(&tmp);

    return err;
}

atf_error_t
atf_text_split(const char *str, const char *delim, atf_list_t *words)
{
    atf_error_t err;
    const char *end;
    const char *iter;

    err = atf_list_init(words);
    if (atf_is_error(err))
        goto err;

    end = str + strlen(str);
    INV(*end == '\0');
    iter = str;
    while (iter < end) {
        const char *ptr;

        INV(iter != NULL);
        ptr = strstr(iter, delim);
        if (ptr == NULL)
            ptr = end;

        INV(ptr >= iter);
        if (ptr > iter) {
            atf_dynstr_t word;

            err = atf_dynstr_init_raw(&word, iter, ptr - iter);
            if (atf_is_error(err))
                goto err_list;

            err = atf_list_append(words, atf_dynstr_fini_disown(&word), true);
            if (atf_is_error(err))
                goto err_list;
        }

        iter = ptr + strlen(delim);
    }

    INV(!atf_is_error(err));
    return err;

err_list:
    atf_list_fini(words);
err:
    return err;
}

atf_error_t
atf_text_to_bool(const char *str, bool *b)
{
    atf_error_t err;

    if (strcasecmp(str, "yes") == 0 ||
        strcasecmp(str, "true") == 0) {
        *b = true;
        err = atf_no_error();
    } else if (strcasecmp(str, "no") == 0 ||
               strcasecmp(str, "false") == 0) {
        *b = false;
        err = atf_no_error();
    } else {
        /* XXX Not really a libc error. */
        err = atf_libc_error(EINVAL, "Cannot convert string '%s' "
                             "to boolean", str);
    }

    return err;
}

atf_error_t
atf_text_to_long(const char *str, long *l)
{
    atf_error_t err;
    char *endptr;
    long tmp;

    errno = 0;
    tmp = strtol(str, &endptr, 10);
    if (str[0] == '\0' || *endptr != '\0')
        err = atf_libc_error(EINVAL, "'%s' is not a number", str);
    else if (errno == ERANGE || (tmp == LONG_MAX || tmp == LONG_MIN))
        err = atf_libc_error(ERANGE, "'%s' is out of range", str);
    else {
        *l = tmp;
        err = atf_no_error();
    }

    return err;
}
