/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 24, 2023.
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
	DB_ENV *dbenv;
	DWORD attrs;
	_TCHAR *tpath;
	int ret;

	dbenv = env == NULL ? NULL : env->dbenv;

	TO_TSTRING(env, path, tpath, ret);
	if (ret != 0)
		return (ret);

	if (dbenv != NULL &&
	    FLD_ISSET(dbenv->verbose, DB_VERB_FILEOPS | DB_VERB_FILEOPS_ALL))
		__db_msg(env, "fileops: stat %s", path);

	RETRY_CHK(
	    ((attrs = GetFileAttributes(tpath)) == (DWORD)-1 ? 1 : 0), ret);
	if (ret == 0) {
		if (isdirp != NULL)
			*isdirp = (attrs & FILE_ATTRIBUTE_DIRECTORY);
	} else
		ret = __os_posix_err(ret);

	FREE_STRING(env, tpath);
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
	int ret;
	BY_HANDLE_FILE_INFORMATION bhfi;
	unsigned __int64 filesize;

	RETRY_CHK((!GetFileInformationByHandle(fhp->handle, &bhfi)), ret);
	if (ret != 0) {
		__db_syserr(env, ret, "GetFileInformationByHandle");
		return (__os_posix_err(ret));
	}

	filesize = ((unsigned __int64)bhfi.nFileSizeHigh << 32) +
	    bhfi.nFileSizeLow;

	/* Return the size of the file. */
	if (mbytesp != NULL)
		*mbytesp = (u_int32_t)(filesize / MEGABYTE);
	if (bytesp != NULL)
		*bytesp = (u_int32_t)(filesize % MEGABYTE);

	/*
	 * The filesystem I/O size is not easily available.  In particular,
	 * the values returned by GetDiskFreeSpace() are not very helpful
	 * (NTFS volumes often report 512B clusters, which are too small to
	 * be a useful default).
	 */
	if (iosizep != NULL)
		*iosizep = DB_DEF_IOSIZE;
	return (0);
}
