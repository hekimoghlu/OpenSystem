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
/*
 * This is an open source non-commercial project. Dear PVS-Studio, please check it.
 * PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
 */

#include <config.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sudoers.h"

#ifdef HAVE_STRLCPY
# define cpy_default	strlcpy
#else
# define cpy_default	sudo_strlcpy
#endif

/*
 * Join a NULL-terminated array of strings using the specified separator
 * char.  If non-NULL, the copy function must have strlcpy-like semantics.
 */
char *
strvec_join(char *const argv[], char sep, size_t (*cpy)(char *, const char *, size_t))
{
    char *dst, *result = NULL;
    char *const *av;
    size_t n, size = 0;
    debug_decl(strvec_join, SUDOERS_DEBUG_UTIL);

    for (av = argv; *av != NULL; av++)
	size += strlen(*av) + 1;
    if (size == 0 || (result = malloc(size)) == NULL) {
	sudo_warnx(U_("%s: %s"), __func__, U_("unable to allocate memory"));
	debug_return_ptr(NULL);
    }

    if (cpy == NULL)
	cpy = cpy_default;
    for (dst = result, av = argv; *av != NULL; av++) {
	n = cpy(dst, *av, size);
	if (n >= size) {
	    sudo_warnx(U_("internal error, %s overflow"), __func__);
	    free(result);
	    debug_return_ptr(NULL);
	}
	dst += n;
	size -= n;
	*dst++ = sep;
	size--;
    }
    dst[-1] = '\0';

    debug_return_str(result);
}
