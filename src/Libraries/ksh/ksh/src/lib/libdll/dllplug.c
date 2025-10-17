/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 8, 2022.
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
 * find and load lib plugin/module library name with optional version ver and dlopen() flags
 * at least one dlopen() is called to initialize dlerror()
 * if path!=0 then library path up to size chars copied to path with trailing 0
 * if name contains a directory prefix then library search is limited to the dir and siblings
 */

extern void*
dllplugin(const char* lib, const char* name, const char* ver, unsigned long rel, unsigned long* cur, int flags, char* path, size_t size)
{
	void*		dll;
	int		err;
	int		hit;
	Dllscan_t*	dls;
	Dllent_t*	dle;

	err = hit = 0;
	for (;;)
	{
		if (dls = dllsopen(lib, name, ver))
		{
			while (dle = dllsread(dls))
			{
				hit = 1;
#if 0
			again:
#endif
				if (dll = dllopen(dle->path, flags|RTLD_GLOBAL|RTLD_PARENT))
				{
					if (!dllcheck(dll, dle->path, rel, cur))
					{
						err = state.error;
						dlclose(dll);
						dll = 0;
						continue;
					}
					if (path && size)
						strlcpy(path, dle->path, size);
					break;
				}
				else
				{
#if 0
					/*
					 * dlopen() should load implicit libraries
					 * this code does that
					 * but it doesn't help on galadriel
					 */

					char*	s;
					char*	e;

					if ((s = dllerror(1)) && (e = strchr(s, ':')))
					{
						*e = 0;
						error(1, "AHA %s implicit", s);
						dll = dllplugin(lib, s, 0, 0, 0, flags, path, size);
						*e = ':';
						if (dll)
						{
							error(1, "AHA implicit %s => %s", s, path);
							goto again;
						}
					}
#endif
					errorf("dll", NiL, 1, "dllplugin: %s dlopen failed: %s", dle->path, dllerror(1));
					err = state.error;
				}
			}
			dllsclose(dls);
		}
		if (hit)
		{
			if (!dll)
				state.error = err;
			return dll;
		}
		if (!lib)
			break;
		lib = 0;
	}
	if (dll = dllopen(name, flags))
	{
		if (!dllcheck(dll, name, rel, cur))
		{
			dlclose(dll);
			dll = 0;
		}
		else if (path && size)
			strlcpy(path, name, size);
	}
	return dll;
}

extern void*
dllplug(const char* lib, const char* name, const char* ver, int flags, char* path, size_t size)
{
	return dllplugin(lib, name, ver, 0, NiL, flags, path, size);
}
