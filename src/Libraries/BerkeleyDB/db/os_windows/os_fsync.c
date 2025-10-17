/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 16, 2024.
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
 * __os_fsync --
 *	Flush a file descriptor.
 */
int
__os_fsync(env, fhp)
	ENV *env;
	DB_FH *fhp;
{
	DB_ENV *dbenv;
	int ret;

	dbenv = env == NULL ? NULL : env->dbenv;

	/*
	 * Do nothing if the file descriptor has been marked as not requiring
	 * any sync to disk.
	 */
	if (F_ISSET(fhp, DB_FH_NOSYNC))
		return (0);

	if (dbenv != NULL && FLD_ISSET(dbenv->verbose, DB_VERB_FILEOPS_ALL))
		__db_msg(env, "fileops: flush %s", fhp->name);

	RETRY_CHK((!FlushFileBuffers(fhp->handle)), ret);
	if (ret != 0) {
		__db_syserr(env, ret, "FlushFileBuffers");
		ret = __os_posix_err(ret);
	}
	return (ret);
}
