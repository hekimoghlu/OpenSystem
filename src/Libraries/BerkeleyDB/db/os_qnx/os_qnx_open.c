/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 31, 2025.
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
#include "db_config.h"

#include "db_int.h"

/*
 * __os_qnx_region_open --
 *	Open a shared memory region file using POSIX shm_open.
 *
 * PUBLIC: #ifdef HAVE_QNX
 * PUBLIC: int __os_qnx_region_open
 * PUBLIC:     __P((ENV *, const char *, int, int, DB_FH **));
 * PUBLIC: #endif
 */
int
__os_qnx_region_open(env, name, oflags, mode, fhpp)
	ENV *env;
	const char *name;
	int oflags, mode;
	DB_FH **fhpp;
{
	DB_FH *fhp;
	int fcntl_flags;
	int ret;

	/*
	 * Allocate the file handle and copy the file name.  We generally only
	 * use the name for verbose or error messages, but on systems where we
	 * can't unlink temporary files immediately, we use the name to unlink
	 * the temporary file when the file handle is closed.
	 *
	 * Lock the ENV handle and insert the new file handle on the list.
	 */
	if ((ret = __os_calloc(env, 1, sizeof(DB_FH), &fhp)) != 0)
		return (ret);
	if ((ret = __os_strdup(env, name, &fhp->name)) != 0)
		goto err;
	if (env != NULL) {
		MUTEX_LOCK(env, env->mtx_env);
		TAILQ_INSERT_TAIL(&env->fdlist, fhp, q);
		MUTEX_UNLOCK(env, env->mtx_env);
		F_SET(fhp, DB_FH_ENVLINK);
	}

	/*
	 * Once we have created the object, we don't need the name
	 * anymore.  Other callers of this will convert themselves.
	 */
	if ((fhp->fd = shm_open(name, oflags, mode)) == -1) {
		ret = __os_posix_err(__os_get_syserr());
err:		(void)__os_closehandle(env, fhp);
		return (ret);
	}

	F_SET(fhp, DB_FH_OPENED);

#ifdef HAVE_FCNTL_F_SETFD
	/* Deny file descriptor access to any child process. */
	if ((fcntl_flags = fcntl(fhp->fd, F_GETFD)) == -1 ||
	    fcntl(fhp->fd, F_SETFD, fcntl_flags | FD_CLOEXEC) == -1) {
		ret = __os_get_syserr();
		__db_syserr(env, ret, "fcntl(F_SETFD)");
		(void)__os_closehandle(env, fhp);
		return (__os_posix_err(ret));
	}
#else
	COMPQUIET(fcntl_flags, 0);
#endif
	F_SET(fhp, DB_FH_OPENED);
	*fhpp = fhp;
	return (0);
}
