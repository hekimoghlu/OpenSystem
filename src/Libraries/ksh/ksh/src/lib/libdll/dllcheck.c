/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 16, 2025.
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
 */

#include "dlllib.h"

/*
 * return plugin version for dll
 * 0 if there is none
 * path!=0 enables library level diagnostics
 */

extern unsigned long
dllversion(void* dll, const char* path)
{
	Dll_plugin_version_f	pvf;

	if (pvf = (Dll_plugin_version_f)dlllook(dll, "plugin_version"))
		return (*pvf)();
	if (path)
	{
		state.error = 1;
		sfsprintf(state.errorbuf, sizeof(state.errorbuf), "plugin_version() not found");
		errorf("dll", NiL, 1, "dllversion: %s: %s", path, state.errorbuf);
	}
	return 0;
}

/*
 * check if dll on path has plugin version >= ver
 * 1 returned on success, 0 on failure
 * path!=0 enables library level diagnostics
 * cur!=0 gets actual version
 */

extern int
dllcheck(void* dll, const char* path, unsigned long ver, unsigned long* cur)
{
	unsigned long		v;

	state.error = 0;
	if (ver || cur)
	{
		v = dllversion(dll, path);
		if (cur)
			*cur = v;
	}
	if (!ver)
		return 1;
	if (!v)
		return 0;
	if (v < ver)
	{
		if (path)
		{
			state.error = 1;
			sfsprintf(state.errorbuf, sizeof(state.errorbuf), "plugin version %lu older than caller %lu", v, ver);
			errorf("dll", NiL, 1, "dllcheck: %s: %s", path, state.errorbuf);
		}
		return 0;
	}
	errorf("dll", NiL, -1, "dllversion: %s: %lu >= %lu", path, v, ver);
	return 1;
}
