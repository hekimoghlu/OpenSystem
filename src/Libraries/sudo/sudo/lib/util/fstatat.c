/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 6, 2022.
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

#include <fcntl.h>
#include <unistd.h>

#include "sudo_compat.h"

#ifndef HAVE_FSTATAT
int
sudo_fstatat(int dfd, const char *path, struct stat *sb, int flag)
{
    int odfd, ret = -1;

    if (dfd == (int)AT_FDCWD) {
	if (ISSET(flag, AT_SYMLINK_NOFOLLOW))
	    return lstat(path, sb);
	else
	    return stat(path, sb);
    }

    /* Save cwd */
    if ((odfd = open(".", O_RDONLY)) == -1)
	goto done;

    if (fchdir(dfd) == -1)
	goto done;

    if (ISSET(flag, AT_SYMLINK_NOFOLLOW))
	ret = lstat(path, sb);
    else
	ret = stat(path, sb);

    /* Restore cwd */
    if (fchdir(odfd) == -1) {
	/* Should not happen */
	ret = -1;
    }

done:
    if (odfd != -1)
	close(odfd);

    return ret;
}
#endif /* HAVE_FSTATAT */
