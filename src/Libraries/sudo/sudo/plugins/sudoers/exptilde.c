/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 28, 2023.
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
#include <grp.h>
#include <pwd.h>

#include "sudoers.h"
#include "pwutil.h"

/*
 * Expand leading tilde in *path, which must be dynamically allocated.
 * Replaces path with the expanded version as needed, freeing the old one.
 * Returns true on success, false on failure.
 */
bool
expand_tilde(char **path, const char *user)
{
    char *npath, *opath = *path;
    char *slash = NULL;
    struct passwd *pw;
    int len;
    debug_decl(expand_tilde, SUDOERS_DEBUG_UTIL);

    switch (*opath++) {
    case '/':
	/* A fully-qualified path, nothing to do. */
	debug_return_bool(true);
    case '~':
	/* See below. */
	break;
    default:
	/* Not a fully-qualified path or one that starts with a tilde. */
	debug_return_bool(false);
    }

    switch (*opath) {
    case '\0':
	/* format: ~ */
	break;
    case '/':
	/* format: ~/foo */
	opath++;
	break;
    default:
	/* format: ~user/foo */
	user = opath;
	slash = strchr(opath, '/');
	if (slash != NULL) {
	    *slash = '\0';
	    opath = slash + 1;
	} else {
	    opath = (char *)"";
	}
    }
    pw = sudo_getpwnam(user);
    if (slash != NULL)
	*slash = '/';
    if (pw == NULL) {
	/* Unknown user. */
	sudo_warnx(U_("unknown user %s"), user);
	debug_return_bool(false);
    }

    len = asprintf(&npath, "%s%s%s", pw->pw_dir, *opath ? "/" : "", opath);
    sudo_pw_delref(pw);
    if (len == -1) {
	sudo_warnx(U_("%s: %s"), __func__, U_("unable to allocate memory"));
	debug_return_bool(false);
    }

    free(*path);
    *path = npath;
    debug_return_bool(true);
}
