/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 24, 2023.
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

#ifdef __linux__
# include <sys/stat.h>
# include <sys/utsname.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sudo_compat.h"
#include "sudo_util.h"

# if defined(__linux__)
/* 
 * On Linux systems that use multi-arch, the actual DSO may be in a
 * machine-specific subdirectory.  If the specified path contains
 * /lib/ or /libexec/, insert a multi-arch directory after it.
 * If sb is non-NULL, stat(2) will be called on the new path, filling in sb.
 * Returns a dynamically allocated string on success and NULL on failure.
 */
char *
sudo_stat_multiarch_v1(const char *path, struct stat *sb)
{
#  if defined(__ILP32__)
    const char *libdirs[] = { "/libx32/", "/lib/", "/libexec/", NULL };
#  elif defined(__LP64__)
    const char *libdirs[] = { "/lib64/", "/lib/", "/libexec/", NULL };
#  else
    const char *libdirs[] = { "/lib32/", "/lib/", "/libexec/", NULL };
#  endif
    const char **lp, *lib, *slash;
    struct utsname unamebuf;
    char *newpath = NULL;
    int len;

    if (uname(&unamebuf) == -1)
	return NULL;

    for (lp = libdirs; *lp != NULL; lp++) {
	/* Replace lib64, lib32, libx32 with lib in new path. */
	const char *newlib = lp == libdirs ? "/lib/" : *lp;

	/* Search for lib dir in path, find the trailing slash. */
	lib = strstr(path, *lp);
	if (lib == NULL)
	    continue;
	slash = lib + strlen(*lp) - 1;

	/* Make sure there isn't already a machine-linux-gnu dir. */
	len = strcspn(slash + 1, "/-");
	if (strncmp(slash + 1 + len, "-linux-gnu/", 11) == 0) {
	    /* Multiarch already present. */
	    break;
	}

	/* Add machine-linux-gnu dir after /lib/ or /libexec/. */
	len = asprintf(&newpath, "%.*s%s%s-linux-gnu%s",
	    (int)(lib - path), path, newlib, unamebuf.machine, slash);
	if (len == -1) {
	    newpath = NULL;
	    break;
	}

	/* If sb was set, use stat(2) to make sure newpath exists. */
	if (sb == NULL || stat(newpath, sb) == 0)
	    break;
	free(newpath);
	newpath = NULL;
    }

    return newpath;
}
#else
char *
sudo_stat_multiarch_v1(const char *path, struct stat *sb)
{
    return NULL;
}
#endif /* __linux__ */
