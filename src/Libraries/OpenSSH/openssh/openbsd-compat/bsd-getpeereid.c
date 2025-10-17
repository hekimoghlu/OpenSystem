/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 16, 2023.
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
#include "includes.h"

#if !defined(HAVE_GETPEEREID)

#include <sys/types.h>
#include <sys/socket.h>

#include <unistd.h>

#if defined(SO_PEERCRED)
int
getpeereid(int s, uid_t *euid, gid_t *gid)
{
	struct ucred cred;
	socklen_t len = sizeof(cred);

	if (getsockopt(s, SOL_SOCKET, SO_PEERCRED, &cred, &len) < 0)
		return (-1);
	*euid = cred.uid;
	*gid = cred.gid;

	return (0);
}
#elif defined(HAVE_GETPEERUCRED)

#ifdef HAVE_UCRED_H
# include <ucred.h>
#endif

int
getpeereid(int s, uid_t *euid, gid_t *gid)
{
	ucred_t *ucred = NULL;

	if (getpeerucred(s, &ucred) == -1)
		return (-1);
	if ((*euid = ucred_geteuid(ucred)) == -1)
		return (-1);
	if ((*gid = ucred_getrgid(ucred)) == -1)
		return (-1);

	ucred_free(ucred);

	return (0);
}
#else
int
getpeereid(int s, uid_t *euid, gid_t *gid)
{
	*euid = geteuid();
	*gid = getgid();

	return (0);
}
#endif /* defined(SO_PEERCRED) */

#endif /* !defined(HAVE_GETPEEREID) */
