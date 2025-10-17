/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 7, 2023.
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
 * convert path to native fs representation in <buf,siz>
 * length of converted path returned
 * if return length >= siz then buf is indeterminate, but another call
 * with siz=length+1 would work
 * if buf==0 then required size is returned
 */

#include <ast.h>

#if _UWIN

extern int	uwin_path(const char*, char*, int);

size_t
pathnative(const char* path, char* buf, size_t siz)
{
	return uwin_path(path, buf, siz);
}

#else

#if __CYGWIN__

extern void	cygwin_conv_to_win32_path(const char*, char*);

size_t
pathnative(const char* path, char* buf, size_t siz)
{
	size_t		n;

	if (!buf || siz < PATH_MAX)
	{
		char	tmp[PATH_MAX];

		cygwin_conv_to_win32_path(path, tmp);
		if ((n = strlen(tmp)) < siz && buf)
			memcpy(buf, tmp, n + 1);
		return n;
	}
	cygwin_conv_to_win32_path(path, buf);
	return strlen(buf);
}

#else

#if __EMX__

size_t
pathnative(const char* path, char* buf, size_t siz)
{
	char*		s;
	size_t		n;

	if (!_fullpath(buf, path, siz))
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
pathnative(const char* path, char* buf, size_t siz)
{
	*buf = 0;
	if (path[1] == ':')
		strlcpy(buf, path, siz);
	else
		unixpath2win(path, 0, buf, siz);
	return strlen(buf);
}

#else

size_t
pathnative(const char* path, char* buf, size_t siz)
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
