/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 4, 2022.
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
 * Glenn Fowler
 * AT&T Research
 *
 * convert native path to posix fs representation in <buf,siz>
 * length of converted path returned
 * if return length >= siz then buf is indeterminate, but another call
 * with siz=length+1 would work
 * if buf==0 then required size is returned
 */

#include <ast.h>

#if _UWIN

#include <uwin.h>

size_t
pathposix(const char* path, char* buf, size_t siz)
{
	return uwin_unpath(path, buf, siz);
}

#else

#if __CYGWIN__

extern void	cygwin_conv_to_posix_path(const char*, char*);

size_t
pathposix(const char* path, char* buf, size_t siz)
{
	size_t		n;

	if (!buf || siz < PATH_MAX)
	{
		char	tmp[PATH_MAX];

		cygwin_conv_to_posix_path(path, tmp);
		if ((n = strlen(tmp)) < siz && buf)
			memcpy(buf, tmp, n + 1);
		return n;
	}
	cygwin_conv_to_posix_path(path, buf);
	return strlen(buf);
}

#else

#if __EMX__ && 0 /* show me the docs */

size_t
pathposix(const char* path, char* buf, size_t siz)
{
	char*		s;
	size_t		n;

	if (!_posixpath(buf, path, siz))
	{
		for (s = buf; *s; s++)
			if (*s == '/')
				*s = '\\';
	}
	else if ((n = strlen(path)) < siz && buf)
		memcpy(buf, path, n + 1);
	return n;
}

#else

#if __INTERIX

#include <interix/interix.h>

size_t
pathposix(const char* path, char *buf, size_t siz)
{
	static const char	pfx[] = "/dev/fs";

	*buf = 0;
	if (!strncasecmp(path, pfx, sizeof(pfx) - 1))
		strlcpy(buf, path, siz);
	else
		winpath2unix(path, PATH_NONSTRICT, buf, siz);
	return strlen(buf);
}

#else

size_t
pathposix(const char* path, char* buf, size_t siz)
{
	size_t		n;

	if ((n = strlen(path)) < siz && buf)
		memcpy(buf, path, n + 1);
	return n;
}

#endif

#endif

#endif

#endif
