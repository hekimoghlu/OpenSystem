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

#include <fcntl.h>
#include <unistd.h>

#include "sudo_compat.h"

#ifndef HAVE_OPENAT
int
sudo_openat(int dfd, const char *path, int flags, mode_t mode)
{
    int fd, odfd;

    if (dfd == AT_FDCWD)
	return open(path, flags, mode);

    /* Save cwd */
    if ((odfd = open(".", O_RDONLY)) == -1)
	return -1;

    if (fchdir(dfd) == -1) {
	close(odfd);
	return -1;
    }

    fd = open(path, flags, mode);

    /* Restore cwd */
    if (fchdir(odfd) == -1) {
	/* Should not happen */
	if (fd != -1) {
	    close(fd);
	    fd = -1;
	}
    }
    close(odfd);

    return fd;
}
#endif /* HAVE_OPENAT */
