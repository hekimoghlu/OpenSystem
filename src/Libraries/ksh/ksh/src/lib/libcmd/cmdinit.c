/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 28, 2025.
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
 * command initialization
 */

#include <cmd.h>
#include <shcmd.h>

int
_cmd_init(int argc, char** argv, Shbltin_t* context, const char* catalog, int flags)
{
	register char*	cp;

	if (argc <= 0)
		return -1;
	if (context)
	{
		if (flags & ERROR_CALLBACK)
		{
			flags &= ~ERROR_CALLBACK;
			flags |= ERROR_NOTIFY;
		}
		else if (flags & ERROR_NOTIFY)
		{
			context->notify = 1;
			flags &= ~ERROR_NOTIFY;
		}
		error_info.flags |= flags;
	}
	if (cp = strrchr(argv[0], '/'))
		cp++;
	else
		cp = argv[0];
	error_info.id = cp;
	if (!error_info.catalog)
		error_info.catalog = catalog;
	opt_info.index = 0;
	return 0;
}

#if __OBSOLETE__ < 20080101

#if defined(__EXPORT__)
#define extern	__EXPORT__
#endif

#undef	cmdinit

extern void
cmdinit(char** argv, Shbltin_t* context, const char* catalog, int flags)
{
	_cmd_init(0, argv, context, catalog, flags);
}

#endif
