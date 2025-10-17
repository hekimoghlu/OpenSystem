/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 27, 2021.
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
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/gen/getpeereid.c,v 1.6 2002/12/16 13:42:13 maxim Exp $");

#include <sys/param.h>
#include <sys/socket.h>
#include <sys/ucred.h>
#include <sys/un.h>

#include <errno.h>
#include <unistd.h>

int
getpeereid(int s, uid_t *euid, gid_t *egid)
{
	struct xucred xuc;
	socklen_t xuclen;
	int error;

	xuclen = sizeof(xuc);
	error = getsockopt(s, 0, LOCAL_PEERCRED, &xuc, &xuclen);
	if (error != 0)
		return (error);
	if (xuc.cr_version != XUCRED_VERSION) {
		errno = EINVAL;
		return (-1);
	}
	*euid = xuc.cr_uid;
	*egid = xuc.cr_gid;
	return (0);
}
