/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 12, 2024.
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

#include <stdlib.h>
#include <grp.h>

#include "sudo_compat.h"
#include "sudo_debug.h"
#include "sudo_fatal.h"
#include "sudo_gettext.h"
#include "sudo_util.h"

/*
 * Parse a comma-separated list of gids into an allocated array of GETGROUPS_T.
 * If a pointer to the base gid is specified, it is stored as the first element
 * in the array.
 * Returns the number of gids in the allocated array.
 */
int
sudo_parse_gids_v1(const char *gidstr, const gid_t *basegid, GETGROUPS_T **gidsp)
{
    int ngids = 0;
    GETGROUPS_T *gids;
    const char *cp = gidstr;
    const char *errstr;
    char *ep;
    debug_decl(sudo_parse_gids, SUDO_DEBUG_UTIL);

    /* Count groups. */
    if (*cp != '\0') {
	ngids++;
	do {
	    if (*cp++ == ',')
		ngids++;
	} while (*cp != '\0');
    }
    /* Base gid is optional. */
    if (basegid != NULL)
	ngids++;
    /* Allocate and fill in array. */
    if (ngids != 0) {
	gids = reallocarray(NULL, ngids, sizeof(GETGROUPS_T));
	if (gids == NULL) {
	    sudo_warnx(U_("%s: %s"), __func__, U_("unable to allocate memory"));
	    debug_return_int(-1);
	}
	ngids = 0;
	if (basegid != NULL)
	    gids[ngids++] = *basegid;
	cp = gidstr;
	do {
	    gids[ngids] = (GETGROUPS_T) sudo_strtoidx(cp, ",", &ep, &errstr);
	    if (errstr != NULL) {
		sudo_warnx(U_("%s: %s"), cp, U_(errstr));
		free(gids);
		debug_return_int(-1);
	    }
	    if (basegid == NULL || gids[ngids] != *basegid)
		ngids++;
	    cp = ep + 1;
	} while (*ep != '\0');
	*gidsp = gids;
    }
    debug_return_int(ngids);
}
