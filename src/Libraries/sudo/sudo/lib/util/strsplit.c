/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 25, 2022.
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

#include "sudo_compat.h"
#include "sudo_debug.h"
#include "sudo_util.h"

/*
 * Like strtok_r but non-destructive and works w/o a NUL terminator.
 * TODO: Optimize by storing current char in a variable ch
 */
const char *
sudo_strsplit_v1(const char *str, const char *endstr, const char *sep, const char **last)
{
    const char *cp, *s;
    debug_decl(sudo_strsplit, SUDO_DEBUG_UTIL);

    /* If no str specified, use last ptr (if any). */
    if (str == NULL)
	str = *last;

    /* Skip leading separator characters. */
    while (str < endstr) {
	for (s = sep; *s != '\0'; s++) {
	    if (*str == *s) {
		str++;
		break;
	    }
	}
	if (*s == '\0')
	    break;
    }

    /* Empty string? */
    if (str >= endstr) {
	*last = endstr;
	debug_return_ptr(NULL);
    }

    /* Scan str until we hit a char from sep. */
    for (cp = str; cp < endstr; cp++) {
	for (s = sep; *s != '\0'; s++) {
	    if (*cp == *s)
		break;
	}
	if (*s != '\0')
	    break;
    }
    *last = cp;
    debug_return_const_ptr(str);
}
