/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 16, 2024.
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
/*
 * nasmlib.c	library routines for the Netwide Assembler
 */

#include "compiler.h"
#include "nasmlib.h"
#include "nctype.h"

#ifndef nasm_stricmp
int nasm_stricmp(const char *s1, const char *s2)
{
    unsigned char c1, c2;
    int d;

    while (1) {
	c1 = nasm_tolower(*s1++);
	c2 = nasm_tolower(*s2++);
	d = c1-c2;

	if (d)
	    return d;
	if (!c1)
	    break;
    }
    return 0;
}
#endif

#ifndef nasm_strnicmp
int nasm_strnicmp(const char *s1, const char *s2, size_t n)
{
    unsigned char c1, c2;
    int d;

    while (n--) {
	c1 = nasm_tolower(*s1++);
	c2 = nasm_tolower(*s2++);
	d = c1-c2;

	if (d)
	    return d;
	if (!c1)
	    break;
    }
    return 0;
}
#endif

int nasm_memicmp(const char *s1, const char *s2, size_t n)
{
    unsigned char c1, c2;
    int d;

    while (n--) {
	c1 = nasm_tolower(*s1++);
	c2 = nasm_tolower(*s2++);
	d = c1-c2;
	if (d)
	    return d;
    }
    return 0;
}

#ifndef nasm_strsep
char *nasm_strsep(char **stringp, const char *delim)
{
        char *s = *stringp;
        char *e;

        if (!s)
                return NULL;

        e = strpbrk(s, delim);
        if (e)
                *e++ = '\0';

        *stringp = e;
        return s;
}
#endif

/* skip leading spaces */
char *nasm_skip_spaces(const char *p)
{
    if (p)
        while (*p && nasm_isspace(*p))
            p++;
    return (char *)p;
}

/* skip leading non-spaces */
char *nasm_skip_word(const char *p)
{
    if (p)
        while (*p && !nasm_isspace(*p))
            p++;
    return (char *)p;
}

/* zap leading spaces with zero */
char *nasm_zap_spaces_fwd(char *p)
{
    if (p)
        while (*p && nasm_isspace(*p))
            *p++ = 0x0;
    return p;
}

/* zap spaces with zero in reverse order */
char *nasm_zap_spaces_rev(char *p)
{
    if (p)
        while (*p && nasm_isspace(*p))
            *p-- = 0x0;
    return p;
}

/* zap leading and trailing spaces */
char *nasm_trim_spaces(char *p)
{
    p = nasm_zap_spaces_fwd(p);
    nasm_zap_spaces_fwd(nasm_skip_word(p));

    return p;
}

/*
 * return the word extracted from a stream
 * or NULL if nothing left
 */
char *nasm_get_word(char *p, char **tail)
{
    char *word = nasm_skip_spaces(p);
    char *next = nasm_skip_word(word);

    if (word && *word) {
        if (*next)
            *next++ = '\0';
    } else
        word = next = NULL;

    /* NOTE: the tail may start with spaces */
    *tail = next;

    return word;
}

/*
 * Extract "opt=val" values from the stream and
 * returns "opt"
 *
 * Exceptions:
 * 1) If "=val" passed the NULL returned though
 *    you may continue handling the tail via "next"
 * 2) If "=" passed the NULL is returned and "val"
 *    is set to NULL as well
 */
char *nasm_opt_val(char *p, char **val, char **next)
{
    char *q, *nxt;

    *val = *next = NULL;

    p = nasm_get_word(p, &nxt);
    if (!p)
        return NULL;

    q = strchr(p, '=');
    if (q) {
        if (q == p)
            p = NULL;
        *q++='\0';
        if (*q) {
            *val = q;
        } else {
            q = nasm_get_word(q + 1, &nxt);
            if (q)
                *val = q;
        }
    } else {
        q = nasm_skip_spaces(nxt);
        if (q && *q == '=') {
            q = nasm_get_word(q + 1, &nxt);
            if (q)
                *val = q;
        }
    }

    *next = nxt;
    return p;
}
