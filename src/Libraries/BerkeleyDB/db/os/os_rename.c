/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 25, 2024.
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
 * __os_rename --
 *	Rename a file.
 *
 * PUBLIC: int __os_rename __P((ENV *,
 * PUBLIC:    const char *, const char *, u_int32_t));
 */
int
__os_rename(env, oldname, newname, silent)
	ENV *env;
	const char *oldname, *newname;
	u_int32_t silent;
{
	DB_ENV *dbenv;
	int ret;

	dbenv = env == NULL ? NULL : env->dbenv;
	if (dbenv != NULL &&
	    FLD_ISSET(dbenv->verbose, DB_VERB_FILEOPS | DB_VERB_FILEOPS_ALL))
		__db_msg(env, "fileops: rename %s to %s", oldname, newname);

	LAST_PANIC_CHECK_BEFORE_IO(env);

	if (DB_GLOBAL(j_rename) != NULL)
		ret = DB_GLOBAL(j_rename)(oldname, newname);
	else
		RETRY_CHK((rename(oldname, newname)), ret);

	/*
	 * If "silent" is not set, then errors are OK and we should not output
	 * an error message.
	 */
	if (ret != 0) {
		if (!silent)
			__db_syserr(
			    env, ret, "rename %s %s", oldname, newname);
		ret = __os_posix_err(ret);
	}
	return (ret);
}
