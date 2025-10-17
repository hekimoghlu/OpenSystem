/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 6, 2023.
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
 * David Korn
 * Glenn Fowler
 * AT&T Research
 */

static const char usage[] =
"[-?\n@(#)$Id: sync (AT&T Research) 2006-10-04 $\n]"
USAGE_LICENSE
"[+NAME?sync - schedule file system updates]"
"[+DESCRIPTION?\bsync\b calls \bsync\b(2), which causes all information "
    "in memory that updates file systems to be scheduled for writing out to "
    "all file systems. The writing, although scheduled, is not necessarily "
    "complete upon return from \bsync\b.]"
"[+?Since \bsync\b(2) has no failure indication, \bsync\b only fails for "
    "option/operand syntax errors, or when \bsync\b(2) does not return, in "
    "which case \bsync\b also does not return.]"
"[+?At minimum \bsync\b should be called before halting the system. Most "
    "systems provide graceful shutdown procedures that include \bsync\b -- "
    "use them if possible.]"
"[+EXIT STATUS?]"
    "{"
        "[+0?\bsync\b(2) returned.]"
        "[+>0?Option/operand syntax error.]"
    "}"
"[+SEE ALSO?\bsync\b(2), \bshutdown\b(8)]"
;

#include <cmd.h>
#include <ls.h>

int
b_sync(int argc, char** argv, Shbltin_t* context)
{
	cmdinit(argc, argv, context, ERROR_CATALOG, 0);
	for (;;)
	{
		switch (optget(argv, usage))
		{
		case ':':
			error(2, "%s", opt_info.arg);
			break;
		case '?':
			error(ERROR_usage(2), "%s", opt_info.arg);
			break;
		}
		break;
	}
	argv += opt_info.index;
	if (error_info.errors || *argv)
		error(ERROR_usage(2), "%s", optusage(NiL));
#if _lib_sync
	sync();
#else
	error(ERROR_usage(2), "failed -- the native system does not provide a sync(2) call");
#endif
	return 0;
}
