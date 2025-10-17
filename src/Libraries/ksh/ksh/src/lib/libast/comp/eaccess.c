/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 30, 2022.
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
#pragma prototyped
/*
 * access() euid/egid implementation
 */

#include <ast.h>
#include <errno.h>
#include <ls.h>

#include "FEATURE/eaccess"

#if _lib_eaccess

NoN(eaccess)

#else

#if defined(__EXPORT__)
#define extern	__EXPORT__
#endif

extern int
eaccess(const char* path, register int flags)
{
#ifdef EFF_ONLY_OK
	return access(path, flags|EFF_ONLY_OK);
#else
#if _lib_euidaccess
	return euidaccess(path, flags);
#else
	register int	mode;
	struct stat	st;

	static int	init;
	static uid_t	ruid;
	static uid_t	euid;
	static gid_t	rgid;
	static gid_t	egid;

	if (!init)
	{
		ruid = getuid();
		euid = geteuid();
		rgid = getgid();
		egid = getegid();
		init = (ruid == euid && rgid == egid) ? 1 : -1;
	}
	if (init > 0 || flags == F_OK)
		return access(path, flags);
	if (stat(path, &st))
		return -1;
	mode = 0;
	if (euid == 0)
	{
		if (!S_ISREG(st.st_mode) || !(flags & X_OK) || (st.st_mode & (S_IXUSR|S_IXGRP|S_IXOTH)))
			return 0;
		goto nope;
	}
	else if (euid == st.st_uid)
	{
		if (flags & R_OK)
			mode |= S_IRUSR;
		if (flags & W_OK)
			mode |= S_IWUSR;
		if (flags & X_OK)
			mode |= S_IXUSR;
	}
	else if (egid == st.st_gid)
	{
#if _lib_getgroups
	setgroup:
#endif
		if (flags & R_OK)
			mode |= S_IRGRP;
		if (flags & W_OK)
			mode |= S_IWGRP;
		if (flags & X_OK)
			mode |= S_IXGRP;
	}
	else
	{
#if _lib_getgroups
		register int	n;

		static int	ngroups = -2;
		static gid_t*	groups; 

		if (ngroups == -2)
		{
			if ((ngroups = getgroups(0, (gid_t*)0)) <= 0)
				ngroups = NGROUPS_MAX;
			if (!(groups = newof(0, gid_t, ngroups + 1, 0)))
				ngroups = -1;
			else
				ngroups = getgroups(ngroups, groups);
		}
		n = ngroups;
		while (--n >= 0)
			if (groups[n] == st.st_gid)
				goto setgroup;
#endif
		if (flags & R_OK)
			mode |= S_IROTH;
		if (flags & W_OK)
			mode |= S_IWOTH;
		if (flags & X_OK)
			mode |= S_IXOTH;
	}
	if ((st.st_mode & mode) == mode)
		return 0;
 nope:
	errno = EACCES;
	return -1;
#endif
#endif
}

#endif
