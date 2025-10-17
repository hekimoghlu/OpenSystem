/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 22, 2024.
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
 * __os_exists --
 *	Return if the file exists.
 */
int
__os_exists(env, path, isdirp)
	ENV *env;
	const char *path;
	int *isdirp;
{
	FileInfo fInfo;
	IFileMgr *pIFileMgr;
	int ret;

	FILE_MANAGER_CREATE(env, pIFileMgr, ret);
	if (ret != 0)
		return (ret);

	ret = 0;
	if (IFILEMGR_Test(pIFileMgr, path) == EFAILED) {
		FILE_MANAGER_ERR(
		    env, pIFileMgr, path, "IFILEMGR_Test", ret);
		goto err;
	}

	if (isdirp != NULL) {
		if (IFILEMGR_GetInfo(pIFileMgr, path, &fInfo) == EFAILED) {
			FILE_MANAGER_ERR(
			    env, pIFileMgr, path, "IFILEMGR_GetInfo", ret);
			goto err;
		}
		*isdirp = fInfo.attrib == _FA_DIR ? 1 : 0;
	}

err:	IFILEMGR_Release(pIFileMgr);

	return (ret);
}

/*
 * __os_ioinfo --
 *	Return file size and I/O size; abstracted to make it easier
 *	to replace.
 */
int
__os_ioinfo(env, path, fhp, mbytesp, bytesp, iosizep)
	ENV *env;
	const char *path;
	DB_FH *fhp;
	u_int32_t *mbytesp, *bytesp, *iosizep;
{
	FileInfo fInfo;
	IFileMgr *pIFileMgr;
	int ret;

	FILE_MANAGER_CREATE(env, pIFileMgr, ret);
	if (ret != 0)
		return (ret);

	if (IFILE_GetInfo(fhp->ifp, &fInfo) != SUCCESS) {
		FILE_MANAGER_ERR(env, pIFileMgr, path, "IFILE_GetInfo", ret);
		goto err;
	}

	/* Return the size of the file. */
	if (mbytesp != NULL)
		*mbytesp = (u_int32_t)(fInfo.dwSize / MEGABYTE);
	if (bytesp != NULL)
		*bytesp = (u_int32_t)(fInfo.dwSize % MEGABYTE);

	/* Default the filesystem I/O size. */
	if (iosizep != NULL)
		*iosizep = DB_DEF_IOSIZE;

err:	IFILEMGR_Release(pIFileMgr);
	return (ret);
}
