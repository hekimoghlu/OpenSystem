/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 25, 2022.
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
#if defined(LIBC_SCCS) && !defined(lint)
static char sccsid[] = "@(#)getmntinfo.c	8.1 (Berkeley) 6/4/93";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/gen/getmntinfo.c,v 1.5 2007/01/09 00:27:54 imp Exp $");

#include <sys/param.h>
#include <sys/ucred.h>
#include <sys/mount.h>
#include <errno.h>
#include <stdlib.h>

struct getmntinfo_vars {
	struct statfs *mntbuf;
	int mntsize;
	long bufsize;
};

/*
 * Return information about mounted filesystems.
 */
static int
getmntinfo_internal(struct getmntinfo_vars *vars, int flags)
{

	if (vars->mntsize <= 0 &&
	    (vars->mntsize = getfsstat(0, 0, MNT_NOWAIT)) < 0) {
		return (0);
	}
	if (vars->bufsize > 0 &&
	    (vars->mntsize =
	     getfsstat(vars->mntbuf, vars->bufsize, flags)) < 0) {
		return (0);
	}
	while (vars->bufsize <= vars->mntsize * sizeof(struct statfs)) {
		if (vars->mntbuf) {
			free(vars->mntbuf);
		}
		vars->bufsize = (vars->mntsize + 1) * sizeof(struct statfs);
		if ((vars->mntbuf =
		     (struct statfs *)malloc(vars->bufsize)) == 0) {
			return (0);
		}
		if ((vars->mntsize =
		     getfsstat(vars->mntbuf, vars->bufsize, flags)) < 0) {
			return (0);
		}
	}
	return (vars->mntsize);
}

/* Legacy version that keeps the buffer around. */
int
getmntinfo(struct statfs **mntbufp, int flags)
{
	static struct getmntinfo_vars vars;
	int rv;

	rv = getmntinfo_internal(&vars, flags);
	/* Unconditional assignment matches legacy behavior. */
	*mntbufp = vars.mntbuf;
	return (rv);
}

/* Thread-safe version where the caller owns the newly-allocated buffer. */
int
getmntinfo_r_np(struct statfs **mntbufp, int flags)
{
	struct getmntinfo_vars vars = { 0 };
	int rv, save_errno;

	if ((rv = getmntinfo_internal(&vars, flags)) != 0) {
		*mntbufp = vars.mntbuf;
	} else {
		save_errno = errno;
		free(vars.mntbuf);
		errno = save_errno;
	}
	return (rv);
}
