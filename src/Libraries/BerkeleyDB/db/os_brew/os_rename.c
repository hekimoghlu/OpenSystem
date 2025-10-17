/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 3, 2024.
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
__os_rename(env, old, new, silent)
	ENV *env;
	const char *old, *new;
	u_int32_t silent;
{
	IFileMgr *pIFileMgr;
	int ret;

	FILE_MANAGER_CREATE(env, pIFileMgr, ret);
	if (ret != 0)
		return (ret);

	LAST_PANIC_CHECK_BEFORE_IO(env);

	if (IFILEMGR_Rename(pIFileMgr, old, new) == SUCCESS)
		ret = 0;
	else
		if (!silent)
			FILE_MANAGER_ERR(env,
			    pIFileMgr, old, "IFILEMGR_Rename", ret);
		else
			ret = __os_posix_err(IFILEMGR_GetLastError(pIFileMgr));

	IFILEMGR_Release(pIFileMgr);
	return (ret);
}
