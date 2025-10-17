/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 19, 2024.
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
 * __os_truncate --
 *	Truncate the file.
 *
 * PUBLIC: int __os_truncate __P((ENV *, DB_FH *, db_pgno_t, u_int32_t));
 */
int
__os_truncate(env, fhp, pgno, pgsize)
	ENV *env;
	DB_FH *fhp;
	db_pgno_t pgno;
	u_int32_t pgsize;
{
	DB_ENV *dbenv;
	off_t offset;
	int ret;

	dbenv = env == NULL ? NULL : env->dbenv;

	/*
	 * Truncate a file so that "pgno" is discarded from the end of the
	 * file.
	 */
	offset = (off_t)pgsize * pgno;

	if (dbenv != NULL &&
	    FLD_ISSET(dbenv->verbose, DB_VERB_FILEOPS | DB_VERB_FILEOPS_ALL))
		__db_msg(env,
		    "fileops: truncate %s to %lu", fhp->name, (u_long)offset);

	LAST_PANIC_CHECK_BEFORE_IO(env);

	if (DB_GLOBAL(j_ftruncate) != NULL)
		ret = DB_GLOBAL(j_ftruncate)(fhp->fd, offset);
	else {
#ifdef HAVE_FTRUNCATE
		RETRY_CHK((ftruncate(fhp->fd, offset)), ret);
#else
		ret = DB_OPNOTSUP;
#endif
	}

	if (ret != 0) {
		__db_syserr(env, ret, "ftruncate: %lu", (u_long)offset);
		ret = __os_posix_err(ret);
	}

	return (ret);
}
