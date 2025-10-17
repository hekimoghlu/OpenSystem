/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 8, 2022.
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
 */
int
__os_rename(env, oldname, newname, silent)
	ENV *env;
	const char *oldname, *newname;
	u_int32_t silent;
{
	DB_ENV *dbenv;
	_TCHAR *toldname, *tnewname;
	int ret;

	dbenv = env == NULL ? NULL : env->dbenv;

	if (dbenv != NULL &&
	    FLD_ISSET(dbenv->verbose, DB_VERB_FILEOPS | DB_VERB_FILEOPS_ALL))
		__db_msg(env, "fileops: rename %s to %s", oldname, newname);

	TO_TSTRING(env, oldname, toldname, ret);
	if (ret != 0)
		return (ret);
	TO_TSTRING(env, newname, tnewname, ret);
	if (ret != 0) {
		FREE_STRING(env, toldname);
		return (ret);
	}

	LAST_PANIC_CHECK_BEFORE_IO(env);

	if (!MoveFile(toldname, tnewname))
		ret = __os_get_syserr();

	if (__os_posix_err(ret) == EEXIST) {
		ret = 0;
#ifndef DB_WINCE
		if (__os_is_winnt()) {
			if (!MoveFileEx(
			    toldname, tnewname, MOVEFILE_REPLACE_EXISTING))
				ret = __os_get_syserr();
		} else
#endif
		{
			/*
			 * There is no MoveFileEx for Win9x/Me/CE, so we have to
			 * do the best we can.  Note that the MoveFile call
			 * above would have succeeded if oldname and newname
			 * refer to the same file, so we don't need to check
			 * that here.
			 */
			(void)DeleteFile(tnewname);
			if (!MoveFile(toldname, tnewname))
				ret = __os_get_syserr();
		}
	}

	FREE_STRING(env, tnewname);
	FREE_STRING(env, toldname);

	if (ret != 0) {
		if (silent == 0)
			__db_syserr(
			    env, ret, "MoveFileEx %s %s", oldname, newname);
		ret = __os_posix_err(ret);
	}

	return (ret);
}
