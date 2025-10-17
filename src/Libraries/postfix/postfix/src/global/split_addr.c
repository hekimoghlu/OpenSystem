/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 4, 2023.
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
/* System library. */

#include <sys_defs.h>
#include <string.h>

#ifdef STRCASECMP_IN_STRINGS_H
#include <strings.h>
#endif

/* Utility library. */

#include <split_at.h>
#include <stringops.h>

/* Global library. */

#include <mail_params.h>
#include <mail_addr.h>
#include <split_addr.h>

/* split_addr_internal - split address with extreme prejudice */

char   *split_addr_internal(char *localpart, const char *delimiter_set)
{
    ssize_t len;

    /*
     * Don't split these, regardless of what the delimiter is.
     */
    if (strcasecmp(localpart, MAIL_ADDR_POSTMASTER) == 0)
	return (0);
    if (strcasecmp(localpart, MAIL_ADDR_MAIL_DAEMON) == 0)
	return (0);
    if (strcasecmp_utf8(localpart, var_double_bounce_sender) == 0)
	return (0);

    /*
     * Backwards compatibility: don't split owner-foo or foo-request.
     */
    if (strchr(delimiter_set, '-') != 0 && var_ownreq_special != 0) {
	if (strncasecmp(localpart, "owner-", 6) == 0)
	    return (0);
	if ((len = strlen(localpart) - 8) > 0
	    && strcasecmp(localpart + len, "-request") == 0)
	    return (0);
    }

    /*
     * Safe to split this address. Do not split the address if the result
     * would have a null localpart.
     */
    if ((len = strcspn(localpart, delimiter_set)) == 0 || localpart[len] == 0) {
	return (0);
    } else {
	localpart[len] = 0;
	return (localpart + len + 1);
    }
}
