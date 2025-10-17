/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 4, 2024.
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

#include <ast.h>

#if _lib_mkdir

NoN(mkdir)

#else

#include <ls.h>
#include <wait.h>
#include <error.h>

int
mkdir(const char* path, mode_t mode)
{
	register int	n;
	char*		av[3];

	static char*	cmd[] = { "/bin/mkdir", "/usr/5bin/mkdir", 0 };


	n = errno;
	if (!access(path, F_OK))
	{
		errno = EEXIST;
		return(-1);
	}
	if (errno != ENOENT) return(-1);
	errno = n;
	av[0] = "mkdir";
	av[1] = path;
	av[2] = 0;
	for (n = 0; n < elementsof(cmd); n++)
		if (procclose(procopen(cmd[n], av, NiL, NiL, 0)) != -1)
			break;
	return(chmod(path, mode));
}

#endif
