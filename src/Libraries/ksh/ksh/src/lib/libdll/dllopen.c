/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 12, 2025.
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
 * at&t research
 */

#include "dlllib.h"

#if 0

/*
 * dlopen() wrapper that properly initializes LIBPATH
 * with the path of the dll to be opened
 *
 * 2009-04-15 -- if ld.so re-checked the env this would work ...
 */

void*
dllopen(const char* name, int mode)
{
	void*		dll;
	Dllinfo_t*	info;
	char*		olibpath;
	char*		path;
	char*		oenv;
	char*		nenv[2];
	char*		dir;
	char*		base;
	int		len;

	if (!environ)
	{
		nenv[0] = nenv[1] = 0;
		environ = nenv;
	}
	info = dllinfo();
	oenv = environ[0];
	olibpath = getenv(info->env);
	if (base = strrchr(name, '/'))
	{
		dir = (char*)name;
		len = ++base - dir;
	}
	else
	{
		dir = "./";
		len = 2;
		base = (char*)name;
	}
	path = sfprints("%-.*s%s%c%s=%-.*s%s%s", len, dir, base, 0, info->env, len, dir, olibpath ? ":" : "", olibpath ? olibpath : "");
	environ[0] = path + strlen(path) + 1;
	state.error = 0;
	dll = dlopen(path, mode);
	if (environ == nenv)
		environ = 0;
	else
		environ[0] = oenv;
	return dll;
}

#else

/*
 * dlopen() wrapper -- waiting for prestidigitaions
 */

void*
dllopen(const char* name, int mode)
{
	state.error = 0;
	return dlopen(name, mode);
}

#endif
