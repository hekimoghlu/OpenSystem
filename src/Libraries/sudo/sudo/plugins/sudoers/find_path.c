/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 16, 2023.
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

#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>

#include "sudoers.h"

/*
 * Check the given command against the specified allowlist (NULL-terminated).
 * On success, rewrites cmnd based on the allowlist and returns true.
 * On failure, returns false.
 */
static bool
cmnd_allowed(char *cmnd, size_t cmnd_size, const char *runchroot,
    struct stat *cmnd_sbp, char * const *allowlist)
{
    const char *cmnd_base;
    char * const *al;
    debug_decl(cmnd_allowed, SUDOERS_DEBUG_UTIL);

    if (!sudo_goodpath(cmnd, runchroot, cmnd_sbp))
	debug_return_bool(false);

    if (allowlist == NULL)
	debug_return_bool(true);	/* nothing to check */

    /* We compare the base names to avoid excessive stat()ing. */
    cmnd_base = sudo_basename(cmnd);

    for (al = allowlist; *al != NULL; al++) {
	const char *base, *path = *al;
	struct stat sb;

	base = sudo_basename(path);
	if (strcmp(cmnd_base, base) != 0)
	    continue;

	if (sudo_goodpath(path, runchroot, &sb) &&
	    sb.st_dev == cmnd_sbp->st_dev && sb.st_ino == cmnd_sbp->st_ino) {
	    /* Overwrite cmnd with safe version from allowlist. */
	    if (strlcpy(cmnd, path, cmnd_size) < cmnd_size)
		debug_return_bool(true);
	}
    }
    debug_return_bool(false);
}

/*
 * This function finds the full pathname for a command and
 * stores it in a statically allocated array, filling in a pointer
 * to the array.  Returns FOUND if the command was found, NOT_FOUND
 * if it was not found, or NOT_FOUND_DOT if it would have been found
 * but it is in '.' and IGNORE_DOT is set.
 * The caller is responsible for freeing the output file.
 */
int
find_path(const char *infile, char **outfile, struct stat *sbp,
    const char *path, const char *runchroot, int ignore_dot,
    char * const *allowlist)
{
    char command[PATH_MAX];
    const char *cp, *ep, *pathend;
    bool found = false;
    bool checkdot = false;
    int len;
    debug_decl(find_path, SUDOERS_DEBUG_UTIL);

    sudo_debug_printf(SUDO_DEBUG_INFO|SUDO_DEBUG_LINENO,
	"resolving %s", infile);

    /*
     * If we were given a fully qualified or relative path
     * there is no need to look at $PATH.
     */
    if (strchr(infile, '/') != NULL) {
	if (strlcpy(command, infile, sizeof(command)) >= sizeof(command)) {
	    errno = ENAMETOOLONG;
	    debug_return_int(NOT_FOUND_ERROR);
	}
	found = cmnd_allowed(command, sizeof(command), runchroot, sbp,
	    allowlist);
	goto done;
    }

    if (path == NULL)
	debug_return_int(NOT_FOUND);

    pathend = path + strlen(path);
    for (cp = sudo_strsplit(path, pathend, ":", &ep); cp != NULL;
	cp = sudo_strsplit(NULL, pathend, ":", &ep)) {

	/*
	 * Search current dir last if it is in PATH.
	 * This will miss sneaky things like using './' or './/' (XXX)
	 */
	if (cp == ep || (*cp == '.' && cp + 1 == ep)) {
	    checkdot = 1;
	    continue;
	}

	/*
	 * Resolve the path and exit the loop if found.
	 */
	len = snprintf(command, sizeof(command), "%.*s/%s",
	    (int)(ep - cp), cp, infile);
	if (len < 0 || len >= ssizeof(command)) {
	    errno = ENAMETOOLONG;
	    debug_return_int(NOT_FOUND_ERROR);
	}
	found = cmnd_allowed(command, sizeof(command), runchroot,
	    sbp, allowlist);
	if (found)
	    break;
    }

    /*
     * Check current dir if dot was in the PATH
     */
    if (!found && checkdot) {
	len = snprintf(command, sizeof(command), "./%s", infile);
	if (len < 0 || len >= ssizeof(command)) {
	    errno = ENAMETOOLONG;
	    debug_return_int(NOT_FOUND_ERROR);
	}
	found = cmnd_allowed(command, sizeof(command), runchroot,
	    sbp, allowlist);
	if (found && ignore_dot)
	    debug_return_int(NOT_FOUND_DOT);
    }

done:
    if (found) {
	sudo_debug_printf(SUDO_DEBUG_INFO|SUDO_DEBUG_LINENO,
	    "found %s", command);
	if ((*outfile = strdup(command)) == NULL)
	    debug_return_int(NOT_FOUND_ERROR);
	debug_return_int(FOUND);
    }
    debug_return_int(NOT_FOUND);
}
