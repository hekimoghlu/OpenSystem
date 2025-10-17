/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 19, 2023.
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
 * __os_mkdir --
 *	Create a directory.
 */
int
__os_mkdir(env, name, mode)
	ENV *env;
	const char *name;
	int mode;
{
	DB_ENV *dbenv;
	_TCHAR *tname;
	int ret;

	dbenv = env == NULL ? NULL : env->dbenv;

	if (dbenv != NULL &&
	    FLD_ISSET(dbenv->verbose, DB_VERB_FILEOPS | DB_VERB_FILEOPS_ALL))
		__db_msg(env, "fileops: mkdir %s", name);

	/* Make the directory, with paranoid permissions. */
	TO_TSTRING(env, name, tname, ret);
	if (ret != 0)
		return (ret);
	RETRY_CHK(!CreateDirectory(tname, NULL), ret);
	FREE_STRING(env, tname);
	if (ret != 0)
		return (__os_posix_err(ret));

	return (ret);
}
