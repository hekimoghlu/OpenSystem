/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 7, 2024.
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

/* extern for regress tests */
bool
matches_env_pattern(const char *pattern, const char *var, bool *full_match)
{
    size_t len, sep_pos;
    bool iswild = false, match = false;
    bool saw_sep = false;
    const char *cp;
    debug_decl(matches_env_pattern, SUDOERS_DEBUG_ENV);

    /* Locate position of the '=' separator in var=value. */
    sep_pos = strcspn(var, "=");

    /* Locate '*' wildcard and compute len. */
    for (cp = pattern; *cp != '\0'; cp++) {
	if (*cp == '*') {
	    iswild = true;
	    break;
	}
    }
    len = (size_t)(cp - pattern);

    if (iswild) {
	/* Match up to the '*' wildcard. */
	if (strncmp(pattern, var, len) == 0) {
	    while (*cp != '\0') {
		if (*cp == '*') {
		    /* Collapse sequential '*'s */
		    do {
			cp++;
		    } while (*cp == '*');
		    /* A '*' at the end of a pattern matches anything. */
		    if (*cp == '\0') {
			match = true;
			break;
		    }
		    /* Keep track of whether we matched an equal sign. */
		    if (*cp == '=')
			saw_sep = true;
		    /* Look for first match of text after the '*' */
		    while ((saw_sep || len != sep_pos) &&
			var[len] != '\0' && var[len] != *cp)
			len++;
		}
		if (var[len] != *cp)
		    break;
		cp++;
		len++;
	    }
	    if (*cp == '\0' && (len == sep_pos || var[len] == '\0'))
		match = true;
	}
    } else {
	if (strncmp(pattern, var, len) == 0 &&
	    (len == sep_pos || var[len] == '\0')) {
	    match = true;
	}
    }
    if (match)
	*full_match = len > sep_pos + 1;
    debug_return_bool(match);
}
