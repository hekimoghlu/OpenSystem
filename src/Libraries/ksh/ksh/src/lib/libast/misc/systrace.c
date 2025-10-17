/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 9, 2024.
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
 * trace systems calls if possible
 */

#include <ast.h>
#include <error.h>
#include <proc.h>
#include <debug.h>

void
systrace(const char* id)
{
	register int	n;
	register char*	out;
	char*		s;
	char		buf[PATH_MAX];
	char*		av[7];
	long		ov[2];

	static char*	trace[] = { "trace", "truss", "strace", "traces" };

	if (!(s = getenv("HOME")))
		return;
	if (!id && !(id = (const char*)error_info.id))
		id = (const char*)trace[0];
	out = buf;
	out += sfsprintf(out, sizeof(buf), "%s/.%s/%s", s, trace[0], id);
	if (access(buf, F_OK))
		return;
	av[1] = trace[0];
	av[2] = "-o";
	av[3] = buf;
	av[4] = "-p";
	av[5] = out + 1;
	av[6] = 0;
	ov[0] = PROC_FD_DUP(open("/dev/null", O_WRONLY), 2, PROC_FD_PARENT|PROC_FD_CHILD);
	ov[1] = 0;
	sfsprintf(out, &buf[sizeof(buf)] - out, ".%d", getpid());
	for (n = 0; n < elementsof(trace); n++)
		if (!procfree(procopen(trace[n], av + 1, NiL, ov, PROC_ARGMOD|PROC_GID|PROC_UID|(n == (elementsof(trace) - 1) ? PROC_CLEANUP : 0))))
		{
			sleep(1);
			break;
		}
}
