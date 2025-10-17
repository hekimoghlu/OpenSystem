/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 24, 2025.
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
 * AT&T Bell Laboratories
 *
 * copy from rfd to wfd (with conditional mmap hacks)
 */

#include <ast.h>
#include <ast_mmap.h>

#if _mmap_worthy > 1

#include <ls.h>

#define MAPSIZE		(1024*256)

#endif

#undef	BUFSIZ
#define BUFSIZ		4096

/*
 * copy n bytes from rfd to wfd
 * actual byte count returned
 * if n<=0 then ``good'' size is used
 */

off_t
astcopy(int rfd, int wfd, off_t n)
{
	register off_t	c;
#ifdef MAPSIZE
	off_t		pos;
	off_t		mapsize;
	char*		mapbuf;
	struct stat	st;
#endif

	static int	bufsiz;
	static char*	buf;

	if (n <= 0 || n >= BUFSIZ * 2)
	{
#if MAPSIZE
		if (!fstat(rfd, &st) && S_ISREG(st.st_mode) && (pos = lseek(rfd, (off_t)0, 1)) != ((off_t)-1))
		{
			if (pos >= st.st_size) return(0);
			mapsize = st.st_size - pos;
			if (mapsize > MAPSIZE) mapsize = (mapsize > n && n > 0) ? n : MAPSIZE;
			if (mapsize >= BUFSIZ * 2 && (mapbuf = (char*)mmap(NiL, mapsize, PROT_READ, MAP_SHARED, rfd, pos)) != ((caddr_t)-1))
			{
				if (write(wfd, mapbuf, mapsize) != mapsize || lseek(rfd, mapsize, 1) == ((off_t)-1)) return(-1);
				munmap((caddr_t)mapbuf, mapsize);
				return(mapsize);
			}
		}
#endif
		if (n <= 0) n = BUFSIZ;
	}
	if (n > bufsiz)
	{
		if (buf) free(buf);
		bufsiz = roundof(n, BUFSIZ);
		if (!(buf = newof(0, char, bufsiz, 0))) return(-1);
	}
	if ((c = read(rfd, buf, (size_t)n)) > 0 && write(wfd, buf, (size_t)c) != c) c = -1;
	return(c);
}
