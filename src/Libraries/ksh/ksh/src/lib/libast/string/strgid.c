/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 14, 2022.
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
 * gid name -> number
 */

#if defined(__STDPP__directive) && defined(__STDPP__hide)
__STDPP__directive pragma pp:hide getgrgid getgrnam getpwnam
#else
#define getgrgid	______getgrgid
#define getgrnam	______getgrnam
#define getpwnam	______getpwnam
#endif

#include <ast.h>
#include <cdt.h>
#include <pwd.h>
#include <grp.h>

#if defined(__STDPP__directive) && defined(__STDPP__hide)
__STDPP__directive pragma pp:nohide getgrgid getgrnam getpwnam
#else
#undef	getgrgid
#undef	getgrnam
#undef	getpwnam
#endif

extern struct group*	getgrgid(gid_t);
extern struct group*	getgrnam(const char*);
extern struct passwd*	getpwnam(const char*);

typedef struct Id_s
{
	Dtlink_t	link;
	int		id;
	char		name[1];
} Id_t;

/*
 * return gid number given gid/uid name
 * gid attempted first, then uid->pw_gid
 * -1 on first error for a given name
 * -2 on subsequent errors for a given name
 */

int
strgid(const char* name)
{
	register Id_t*		ip;
	register struct group*	gr;
	register struct passwd*	pw;
	int			id;
	char*			e;

	static Dt_t*		dict;
	static Dtdisc_t		disc;

	if (!dict)
	{
		disc.key = offsetof(Id_t, name);
		dict = dtopen(&disc, Dtset);
	}
	else if (ip = (Id_t*)dtmatch(dict, name))
		return ip->id;
	if (gr = getgrnam(name))
		id = gr->gr_gid;
	else if (pw = getpwnam(name))
		id = pw->pw_gid;
	else
	{
		id = strtol(name, &e, 0);
#if _WINIX
		if (!*e)
		{
			if (!getgrgid(id))
				id = -1;
		}
		else if (!streq(name, "sys"))
			id = -1;
		else if (gr = getgrnam("Administrators"))
			id = gr->gr_gid;
		else if (pw = getpwnam("Administrator"))
			id = pw->pw_gid;
		else
			id = -1;
#else
		if (*e || !getgrgid(id))
			id = -1;
#endif
	}
	if (dict && (ip = newof(0, Id_t, 1, strlen(name))))
	{
		strcpy(ip->name, name);
		ip->id = id >= 0 ? id : -2;
		dtinsert(dict, ip);
	}
	return id;
}
