/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 9, 2024.
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
#include <string.h>
#include <errno.h>

#include "sudoers.h"

/*
 * Verify that path is a normal file and executable by root.
 */
bool
sudo_goodpath(const char *path, const char *runchroot, struct stat *sbp)
{
    bool ret = false;
    debug_decl(sudo_goodpath, SUDOERS_DEBUG_UTIL);

    if (path != NULL) {
	char pathbuf[PATH_MAX];
	struct stat sb;

	if (runchroot != NULL) {
	    const int len =
		snprintf(pathbuf, sizeof(pathbuf), "%s%s", runchroot, path);
	    if (len >= ssizeof(pathbuf)) {
		errno = ENAMETOOLONG;
		goto done;
	    }
	    path = pathbuf; // -V507
	}
	if (sbp == NULL)
	    sbp = &sb;

	if (stat(path, sbp) == 0) {
	    /* Make sure path describes an executable regular file. */
	    if (S_ISREG(sbp->st_mode) && ISSET(sbp->st_mode, S_IXUSR|S_IXGRP|S_IXOTH))
		ret = true;
	    else
		errno = EACCES;
	}
    }
done:
    debug_return_bool(ret);
}
