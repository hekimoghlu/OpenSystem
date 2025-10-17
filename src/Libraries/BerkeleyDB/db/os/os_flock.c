/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 22, 2025.
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
 * __os_fdlock --
 *	Acquire/release a lock on a byte in a file.
 *
 * PUBLIC: int __os_fdlock __P((ENV *, DB_FH *, off_t, int, int));
 */
int
__os_fdlock(env, fhp, offset, acquire, nowait)
	ENV *env;
	DB_FH *fhp;
	int acquire, nowait;
	off_t offset;
{
#ifdef HAVE_FCNTL
	DB_ENV *dbenv;
	struct flock fl;
	int ret, t_ret;

	dbenv = env == NULL ? NULL : env->dbenv;

	DB_ASSERT(env, F_ISSET(fhp, DB_FH_OPENED) && fhp->fd != -1);

	if (dbenv != NULL && FLD_ISSET(dbenv->verbose, DB_VERB_FILEOPS_ALL))
		__db_msg(env,
		    "fileops: flock %s %s offset %lu",
		    fhp->name, acquire ? "acquire": "release", (u_long)offset);

	fl.l_start = offset;
	fl.l_len = 1;
	fl.l_type = acquire ? F_WRLCK : F_UNLCK;
	fl.l_whence = SEEK_SET;

	RETRY_CHK_EINTR_ONLY(
	    (fcntl(fhp->fd, nowait ? F_SETLK : F_SETLKW, &fl)), ret);

	if (ret == 0)
		return (0);

	if ((t_ret = __os_posix_err(ret)) != EACCES && t_ret != EAGAIN)
		__db_syserr(env, ret, "fcntl");
	return (t_ret);
#else
	COMPQUIET(fhp, NULL);
	COMPQUIET(acquire, 0);
	COMPQUIET(nowait, 0);
	COMPQUIET(offset, 0);
	__db_syserr(env, DB_OPNOTSUP, "advisory file locking unavailable");
	return (DB_OPNOTSUP);
#endif
}
