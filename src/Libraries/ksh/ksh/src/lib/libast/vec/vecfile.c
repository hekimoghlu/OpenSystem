/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 19, 2023.
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
 * string vector load support
 */

#include <ast.h>
#include <ls.h>
#include <vecargs.h>

/*
 * load a string vector from lines in file
 */

char**
vecfile(const char* file)
{
	register int	n;
	register char*	buf;
	register char**	vec;
	int		fd;
	struct stat	st;

	vec = 0;
	if ((fd = open(file, O_RDONLY|O_cloexec)) >= 0)
	{
		if (!fstat(fd, &st) && S_ISREG(st.st_mode) && (n = st.st_size) > 0 && (buf = newof(0, char, n + 1, 0)))
		{
			if (read(fd, buf, n) == n)
			{
				buf[n] = 0;
				vec = vecload(buf);
			}
			if (!vec) free(buf);
		}
		close(fd);
	}
	return(vec);
}
