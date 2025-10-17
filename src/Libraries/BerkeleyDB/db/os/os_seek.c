/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 18, 2022.
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
 * __os_seek --
 *	Seek to a page/byte offset in the file.
 *
 * PUBLIC: int __os_seek __P((ENV *,
 * PUBLIC:      DB_FH *, db_pgno_t, u_int32_t, u_int32_t));
 */
int
__os_seek(env, fhp, pgno, pgsize, relative)
	ENV *env;
	DB_FH *fhp;
	db_pgno_t pgno;
	u_int32_t pgsize;
	u_int32_t relative;
{
	DB_ENV *dbenv;
	off_t offset;
	int ret;

	dbenv = env == NULL ? NULL : env->dbenv;

	DB_ASSERT(env, F_ISSET(fhp, DB_FH_OPENED) && fhp->fd != -1);

#if defined(HAVE_STATISTICS)
	++fhp->seek_count;
#endif

	offset = (off_t)pgsize * pgno + relative;

	if (dbenv != NULL && FLD_ISSET(dbenv->verbose, DB_VERB_FILEOPS_ALL))
		__db_msg(env,
		    "fileops: seek %s to %lu", fhp->name, (u_long)offset);

	if (DB_GLOBAL(j_seek) != NULL)
		ret = DB_GLOBAL(j_seek)(fhp->fd, offset, SEEK_SET);
	else
		RETRY_CHK((lseek(
		    fhp->fd, offset, SEEK_SET) == -1 ? 1 : 0), ret);

	if (ret == 0) {
		fhp->pgsize = pgsize;
		fhp->pgno = pgno;
		fhp->offset = relative;
	} else {
		__db_syserr(env, ret,
		    "seek: %lu: (%lu * %lu) + %lu", (u_long)offset,
		    (u_long)pgno, (u_long)pgsize, (u_long)relative);
		ret = __os_posix_err(ret);
	}

	return (ret);
}
