/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 25, 2024.
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

#ifndef HAVE_DUP3

#include <errno.h>
#include <fcntl.h>
#include <unistd.h>

#include "sudo_compat.h"

int
sudo_dup3(int oldd, int newd, int flags)
{
    int oflags;

    if (oldd == newd) {
	errno = EINVAL;
	return -1;
    }

    if (dup2(oldd, newd) == -1)
	return -1;

    oflags = fcntl(newd, F_GETFL, 0);
    if (oflags == -1)
	goto bad;

    if (ISSET(flags, O_NONBLOCK)) {
	if (!ISSET(oflags, O_NONBLOCK)) {
	    SET(oflags, O_NONBLOCK);
	    if (fcntl(newd, F_SETFL, oflags) == -1)
		goto bad;
	}
    } else {
	if (ISSET(oflags, O_NONBLOCK)) {
	    CLR(oflags, O_NONBLOCK);
	    if (fcntl(newd, F_SETFL, oflags) == -1)
		goto bad;
	}
    }
    if (ISSET(flags, O_CLOEXEC)) {
	if (fcntl(newd, F_SETFD, FD_CLOEXEC) == -1)
	    goto bad;
    }
    return 0;
bad:
    close(newd);
    return -1;
}

#endif /* HAVE_DUP3 */
