/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 9, 2025.
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
 * This is an open source non-commercial project. Dear PVS-Studio, please check it.
 * PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
 */

#include <config.h>

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "sudo_compat.h"
#include "sudo_debug.h"
#include "sudo_util.h"

/* Trivial reference-counted strings. */
struct rcstr {
    int refcnt;
    char str[1];	/* actually bigger */
};

/*
 * Allocate a reference-counted string and copy src to it.
 * Returns the newly-created string with a refcnt of 1.
 */
char *
sudo_rcstr_dup(const char *src)
{
    size_t len = strlen(src);
    char *dst;
    debug_decl(sudo_rcstr_dup, SUDO_DEBUG_UTIL);

    dst = sudo_rcstr_alloc(len);
    if (dst != NULL) {
        memcpy(dst, src, len);
        dst[len] = '\0';
    }
    debug_return_ptr(dst);
}

char *
sudo_rcstr_alloc(size_t len)
{
    struct rcstr *rcs;
    debug_decl(sudo_rcstr_dup, SUDO_DEBUG_UTIL);

    /* Note: sizeof(struct rcstr) includes space for the NUL */
    rcs = malloc(sizeof(struct rcstr) + len);
    if (rcs == NULL)
	return NULL;

    rcs->refcnt = 1;
    rcs->str[0] = '\0';
    /* cppcheck-suppress memleak */
    debug_return_ptr(rcs->str); // -V773
}

char *
sudo_rcstr_addref(const char *s)
{
    struct rcstr *rcs;
    debug_decl(sudo_rcstr_dup, SUDO_DEBUG_UTIL);

    if (s == NULL)
	debug_return_ptr(NULL);

    rcs = __containerof((const void *)s, struct rcstr, str);
    rcs->refcnt++;
    debug_return_ptr(rcs->str);
}

void
sudo_rcstr_delref(const char *s)
{
    struct rcstr *rcs;
    debug_decl(sudo_rcstr_dup, SUDO_DEBUG_UTIL);

    if (s != NULL) {
	rcs = __containerof((const void *)s, struct rcstr, str);
	if (--rcs->refcnt == 0) {
	    rcs->str[0] = '\0';
	    free(rcs);
	}
    }
    debug_return;
}
