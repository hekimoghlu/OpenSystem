/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 29, 2022.
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

#if !__DARWIN_ONLY_64_BIT_INO_T

#include <sys/param.h>
#include <sys/ucred.h>
#include <sys/mount.h>
#include <stdlib.h>

/*
 * Return information about mounted filesystems.
 */
int
getmntinfo64(mntbufp, flags)
	struct statfs64 **mntbufp;
	int flags;
{
	static struct statfs64 *mntbuf;
	static int mntsize;
	static long bufsize;

	if (mntsize <= 0 && (mntsize = getfsstat64(0, 0, MNT_NOWAIT)) < 0)
		return (0);
	if (bufsize > 0 && (mntsize = getfsstat64(mntbuf, bufsize, flags)) < 0)
		return (0);
	while (bufsize <= mntsize * sizeof(struct statfs64)) {
		if (mntbuf)
			free(mntbuf);
		bufsize = (mntsize + 1) * sizeof(struct statfs64);
		if ((mntbuf = (struct statfs64 *)malloc(bufsize)) == 0)
			return (0);
		if ((mntsize = getfsstat64(mntbuf, bufsize, flags)) < 0)
			return (0);
	}
	*mntbufp = mntbuf;
	return (mntsize);
}
#endif
