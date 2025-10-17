/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 25, 2022.
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

typedef void* (*Dll_lib_f)(const char*, void*, const char*);

typedef struct Dll_lib_s
{
	struct Dll_lib_s*	next;
	Dll_lib_f		libf;
	char*			path;
	char			base[1];
} Dll_lib_t;

/*
 * split <name,base,type,opts> from name into names
 */

Dllnames_t*
dllnames(const char* id, const char* name, Dllnames_t* names)
{
	char*	s;
	char*	t;
	char*	b;
	char*	e;
	size_t	n;

	n = strlen(id);
	if (strneq(name, id, n) && (streq(name + n, "_s") || streq(name + n, "_t")))
		return 0;
	if (!names)
	{
		s = fmtbuf(sizeof(Dllnames_t*) + sizeof(names) - 1);
		if (n = (s - (char*)0) % sizeof(names))
			s += sizeof(names) - n;
		names = (Dllnames_t*)s;
	}

	/*
	 * determine the base name
	 */

	if ((s = strrchr(name, '/')) || (s = strrchr(name, '\\')))
		s++;
	else
		s = (char*)name;
	if (strneq(s, "lib", 3))
		s += 3;
	b = names->base = names->data;
	e = b + sizeof(names->data) - 1;
	t = s;
	while (b < e && *t && *t != '.' && *t != '-' && *t != ':')
		*b++ = *t++;
	*b++ = 0;

	/*
	 * determine the optional type
	 */

	if (t = strrchr(s, ':'))
	{
		names->name = b;
		while (b < e && s < t)
			*b++ = *s++;
		*b++ = 0;
		names->type = b;
		while (b < e && *++t)
			*b++ = *t;
		*b++ = 0;
	}
	else
	{
		names->name = (char*)name;
		names->type = 0;
	}
	*(names->path = b) = 0;
	names->opts = 0;
	names->id = (char*)id;
	return names;
}

/*
 * return method pointer for <id,version> in names
 */

void*
dll_lib(Dllnames_t* names, unsigned long version, Dllerror_f dllerrorf, void* disc)
{
	void*			dll;
	Dll_lib_t*		lib;
	Dll_lib_f		libf;
	ssize_t			n;
	char			sym[64];

	static Dll_lib_t*	loaded;

	if (!names)
		return 0;

	/*
	 * check if plugin already loaded
	 */

	for (lib = loaded; lib; lib = lib->next)
		if (streq(names->base, lib->base))
		{
			libf = lib->libf;
			goto init;
		}

	/*
	 * load
	 */

	if (!(dll = dllplugin(names->id, names->name, NiL, version, NiL, RTLD_LAZY, names->path, names->data + sizeof(names->data) - names->path)) && (streq(names->name, names->base) || !(dll = dllplugin(names->id, names->base, NiL, version, NiL, RTLD_LAZY, names->path, names->data + sizeof(names->data) - names->path))))
	{
		if (dllerrorf)
			(*dllerrorf)(NiL, disc, 2, "%s: library not found", names->name);
		else
			errorf("dll", NiL, -1, "dll_lib: %s version %lu library not found", names->name, version);
		return 0;
	}

	/*
	 * init
	 */

	sfsprintf(sym, sizeof(sym), "%s_lib", names->id);
	if (!(libf = (Dll_lib_f)dlllook(dll, sym)))
	{
		if (dllerrorf)
			(*dllerrorf)(NiL, disc, 2, "%s: %s: initialization function not found in library", names->path, sym);
		else
			errorf("dll", NiL, -1, "dll_lib: %s version %lu initialization function %s not found in library", names->name, version, sym);
		return 0;
	}

	/*
	 * add to the loaded list
	 */

	if (lib = newof(0, Dll_lib_t, 1, (n = strlen(names->base)) + strlen(names->path) + 1))
	{
		lib->libf = libf;
		strcpy(lib->base, names->base);
		strcpy(lib->path = lib->base + n + 1, names->path);
		lib->next = loaded;
		loaded = lib;
		errorf("dll", NiL, -1, "dll_lib: %s version %lu loaded from %s", names->name, version, lib->path);
	}
 init:
	return (*libf)(names->path, disc, names->type);
}

/*
 * return method pointer for <id,name,version>
 */

void*
dllmeth(const char* id, const char* name, unsigned long version)
{
	Dllnames_t	names;

	return dll_lib(dllnames(id, name, &names), version, 0, 0);
}
