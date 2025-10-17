/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 16, 2023.
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
 *
 * PUBLIC: int __os_mkdir __P((ENV *, const char *, int));
 */
int
__os_mkdir(env, name, mode)
	ENV *env;
	const char *name;
	int mode;
{
	DB_ENV *dbenv;
	int ret;

	dbenv = env == NULL ? NULL : env->dbenv;
	if (dbenv != NULL &&
	    FLD_ISSET(dbenv->verbose, DB_VERB_FILEOPS | DB_VERB_FILEOPS_ALL))
		__db_msg(env, "fileops: mkdir %s", name);

	/* Make the directory, with paranoid permissions. */
#if defined(HAVE_VXWORKS)
	RETRY_CHK((mkdir(CHAR_STAR_CAST name)), ret);
#else
	RETRY_CHK((mkdir(name, DB_MODE_700)), ret);
#endif
	if (ret != 0)
		return (__os_posix_err(ret));

	/* Set the absolute permissions, if specified. */
#if !defined(HAVE_VXWORKS)
	if (mode != 0) {
		RETRY_CHK((chmod(name, mode)), ret);
		if (ret != 0)
			ret = __os_posix_err(ret);
	}
#endif
	return (ret);
}
