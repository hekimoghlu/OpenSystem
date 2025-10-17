/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 3, 2024.
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
 * uid number -> name
 */

#if defined(__STDPP__directive) && defined(__STDPP__hide)
__STDPP__directive pragma pp:hide getpwuid
#else
#define getpwuid	______getpwuid
#endif

#include <ast.h>
#include <cdt.h>
#include <pwd.h>

#if defined(__STDPP__directive) && defined(__STDPP__hide)
__STDPP__directive pragma pp:nohide getpwuid
#else
#undef	getpwuid
#endif

extern struct passwd*	getpwuid(uid_t);

typedef struct Id_s
{
	Dtlink_t	link;
	int		id;
	char		name[1];
} Id_t;

/*
 * return uid name given uid number
 */

char*
fmtuid(int uid)
{
	register Id_t*		ip;
	register char*		name;
	register struct passwd*	pw;
	int			z;

	static Dt_t*		dict;
	static Dtdisc_t		disc;

	if (!dict)
	{
		disc.key = offsetof(Id_t, id);
		disc.size = sizeof(int);
		dict = dtopen(&disc, Dtset);
	}
	else if (ip = (Id_t*)dtmatch(dict, &uid))
		return ip->name;
	if (pw = getpwuid(uid))
	{
		name = pw->pw_name;
#if _WINIX
		if (streq(name, "Administrator"))
			name = "root";
#endif
	}
	else if (uid == 0)
		name = "root";
	else
	{
		name = fmtbuf(z = sizeof(uid) * 3 + 1);
		sfsprintf(name, z, "%I*d", sizeof(uid), uid);
	}
	if (dict && (ip = newof(0, Id_t, 1, strlen(name))))
	{
		ip->id = uid;
		strcpy(ip->name, name);
		dtinsert(dict, ip);
		return ip->name;
	}
	return name;
}
