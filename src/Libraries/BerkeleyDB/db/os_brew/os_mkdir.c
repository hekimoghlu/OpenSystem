/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 16, 2025.
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
	IFileMgr *ifmp;
	int ret;

	COMPQUIET(mode, 0);

	FILE_MANAGER_CREATE(env, ifmp, ret);
	if (ret != 0)
		return (ret);

	if (IFILEMGR_MkDir(ifmp, name) == SUCCESS)
		ret = 0;
	else
		FILE_MANAGER_ERR(env, ifmp, name, "IFILEMGR_MkDir", ret);

	IFILEMGR_Release(ifmp);

	return (ret);
}
