/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 7, 2023.
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
 * __os_dirlist --
 *	Return a list of the files in a directory.
 */
int
__os_dirlist(env, dir, returndir, namesp, cntp)
	ENV *env;
	const char *dir;
	int returndir, *cntp;
	char ***namesp;
{
	FileInfo fi;
	IFileMgr *pIFileMgr;
	int arraysz, cnt, ret;
	char *filename, *p, **names;

	FILE_MANAGER_CREATE(env, pIFileMgr, ret);
	if (ret != 0)
		return (ret);

	if ((ret = IFILEMGR_EnumInit(pIFileMgr, dir, FALSE)) != SUCCESS) {
		IFILEMGR_Release(pIFileMgr);
		__db_syserr(env, ret, "IFILEMGR_EnumInit");
		return (__os_posix_err(ret));
	}

	names = NULL;
	arraysz = cnt = 0;
	while (IFILEMGR_EnumNext(pIFileMgr, &fi) != FALSE) {
		if (++cnt >= arraysz) {
			arraysz += 100;
			if ((ret = __os_realloc(env,
			    (u_int)arraysz * sizeof(char *), &names)) != 0)
				goto nomem;
		}
		for (filename = fi.szName;
		    (p = strchr(filename, '\\')) != NULL; filename = p + 1)
			;
		for (; (p = strchr(filename, '/')) != NULL; filename = p + 1)
			;
		if ((ret = __os_strdup(env, filename, &names[cnt - 1])) != 0)
			goto nomem;
	}
	IFILEMGR_Release(pIFileMgr);

	*namesp = names;
	*cntp = cnt;
	return (ret);

nomem:	if (names != NULL)
		__os_dirfree(env, names, cnt);
	IFILEMGR_Release(pIFileMgr);

	COMPQUIET(returndir, 0);

	return (ret);
}

/*
 * __os_dirfree --
 *	Free the list of files.
 */
void
__os_dirfree(env, names, cnt)
	ENV *env;
	char **names;
	int cnt;
{
	while (cnt > 0)
		__os_free(env, names[--cnt]);
	__os_free(env, names);
}
