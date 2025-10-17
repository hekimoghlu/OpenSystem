/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 25, 2024.
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
#pragma prototyped

#include <ast.h>

#if _lib_setpgid

NoN(setpgid)

#else

#include <error.h>

#ifndef ENOSYS
#define ENOSYS		EINVAL
#endif

#if _lib_setpgrp2
#define setpgrp		setpgrp2
#else
#if _lib_BSDsetpgrp
#define _lib_setpgrp2	1
#define setpgrp		BSDsetpgrp
#else
#if _lib_wait3
#define	_lib_setpgrp2	1
#endif
#endif
#endif

#if _lib_setpgrp2
extern int		setpgrp(int, int);
#else
extern int		setpgrp(void);
#endif

/*
 * set process group id
 */

int
setpgid(pid_t pid, pid_t pgid)
{
#if _lib_setpgrp2
	return(setpgrp(pid, pgid));
#else
#if _lib_setpgrp
	int	caller = getpid();

	if ((pid == 0 || pid == caller) && (pgid == 0 || pgid == caller))
		return(setpgrp());
	errno = EINVAL;
#else
	errno = ENOSYS;
#endif
	return(-1);
#endif
}

#endif
