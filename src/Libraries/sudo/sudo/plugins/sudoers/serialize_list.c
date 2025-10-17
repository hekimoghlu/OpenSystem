/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 18, 2025.
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

/*
 * Convert struct list_members to a comma-separated string with
 * the given variable name.  Escapes backslashes and commas.
 */
char *
serialize_list(const char *varname, struct list_members *members)
{
    struct list_member *lm, *next;
    size_t len, result_size;
    char *cp, *result;
    debug_decl(serialize_list, SUDOERS_DEBUG_PLUGIN);

    result_size = strlen(varname) + 1;
    SLIST_FOREACH(lm, members, entries) {
	for (cp = lm->value; *cp != '\0'; cp++) {
	    result_size++;
	    if (*cp == '\\' || *cp == ',')
		result_size++;
	}
	result_size++;
    }
    if ((result = malloc(result_size)) == NULL)
	goto bad;
    /* No need to check len for overflow here. */
    len = strlcpy(result, varname, result_size);
    result[len++] = '=';
    SLIST_FOREACH_SAFE(lm, members, entries, next) {
	for (cp = lm->value; *cp != '\0'; cp++) {
	    bool escape = (*cp == '\\' || *cp == ',');
	    if (len + 1 + escape >= result_size) {
		sudo_warnx(U_("internal error, %s overflow"), __func__);
		goto bad;
	    }
	    if (escape)
		result[len++] = '\\';
	    result[len++] = *cp;
	}
	if (next != NULL) {
	    if (len + 1 >= result_size) {
		sudo_warnx(U_("internal error, %s overflow"), __func__);
		goto bad;
	    }
	    result[len++] = ',';
	}
	result[len] = '\0';
    }
    debug_return_str(result);
bad:
    free(result);
    debug_return_str(NULL);
}
