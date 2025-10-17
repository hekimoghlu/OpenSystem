/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 14, 2023.
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
/*
 * This module is used to "verify" password entries by chpass(1) and
 * pwd_mkdb(8).
 */

#include <sys/param.h>

#include <err.h>
#include <errno.h>
#include <pwd.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

#include "pw_scan.h"

/*
 * Some software assumes that IDs are short.  We should emit warnings
 * for id's which cannot be stored in a short, but we are more liberal
 * by default, warning for IDs greater than USHRT_MAX.
 *
 * If pw_big_ids_warning is -1 on entry to pw_scan(), it will be set based
 * on the existence of PW_SCAN_BIG_IDS in the environment.
 *
 * It is believed all baseline system software that can not handle the
 * normal ID sizes is now gone so pw_big_ids_warning is disabled for now.
 * But the code has been left in place in case end-users want to re-enable
 * it and/or for the next time the ID sizes get bigger but pieces of the
 * system lag behind.
 */
static int	pw_big_ids_warning = 0;

void
__pw_initpwd(struct passwd *pwd)
{
	static char nul[] = "";

	memset(pwd, 0, sizeof(*pwd));
	pwd->pw_uid = (uid_t)-1;  /* Considered least likely to lead to */
	pwd->pw_gid = (gid_t)-1;  /* a security issue.                  */
	pwd->pw_name = nul;
	pwd->pw_passwd = nul;
	pwd->pw_class = nul;
	pwd->pw_gecos = nul;
	pwd->pw_dir = nul;
	pwd->pw_shell = nul;
}

int
__pw_scan(char *bp, struct passwd *pw, int flags)
{
#ifdef __APPLE__
	id_t id;
#else
	uid_t id;
#endif /* __APPLE__ */
	int root;
	char *ep, *p, *sh;
#ifdef __APPLE__
	long temp;
#else
	unsigned long temp;
#endif /* __APPLE__ */

	if (pw_big_ids_warning == -1)
		pw_big_ids_warning = getenv("PW_SCAN_BIG_IDS") == NULL ? 1 : 0;

#ifndef __APPLE__
	pw->pw_fields = 0;
#endif /* !__APPLE__ */
	if (!(pw->pw_name = strsep(&bp, ":")))		/* login */
		goto fmt;
	root = !strcmp(pw->pw_name, "root");
#ifndef __APPLE__
	if (pw->pw_name[0] && (pw->pw_name[0] != '+' || pw->pw_name[1] == '\0'))
		pw->pw_fields |= _PWF_NAME;
#endif /* !__APPLE__ */

	if (!(pw->pw_passwd = strsep(&bp, ":")))	/* passwd */
		goto fmt;
#ifndef __APPLE__
	if (pw->pw_passwd[0])
		pw->pw_fields |= _PWF_PASSWD;
#endif /* !__APPLE__ */

	if (!(p = strsep(&bp, ":")))			/* uid */
		goto fmt;
	if (p[0])
#ifdef __APPLE__
		/* nothing */;
#else /* !__APPLE__ */
		pw->pw_fields |= _PWF_UID;
#endif /* !__APPLE__ */
	else {
		if (pw->pw_name[0] != '+' && pw->pw_name[0] != '-') {
			if (flags & _PWSCAN_WARN)
				warnx("no uid for user %s", pw->pw_name);
			return (0);
		}
	}
	errno = 0;
#ifdef __APPLE__
	temp = strtol(p, &ep, 10);
	if ((temp == LONG_MAX && errno == ERANGE) || temp > UID_MAX) {
#else /* !__APPLE__ */
	temp = strtoul(p, &ep, 10);
	if ((temp == ULONG_MAX && errno == ERANGE) || temp > UID_MAX) {
#endif /* !__APPLE__ */
		if (flags & _PWSCAN_WARN)
			warnx("%s > max uid value (%u)", p, UID_MAX);
		return (0);
	}
#ifdef __APPLE__
	id = (id_t)temp;
#else /* !__APPLE__ */
	id = temp;
#endif /* !__APPLE__ */
	if (*ep != '\0') {
		if (flags & _PWSCAN_WARN)
			warnx("%s uid is incorrect", p);
		return (0);
	}
	if (root && id) {
		if (flags & _PWSCAN_WARN)
			warnx("root uid should be 0");
		return (0);
	}
	if (flags & _PWSCAN_WARN && pw_big_ids_warning && id > USHRT_MAX) {
		warnx("%s > recommended max uid value (%u)", p, USHRT_MAX);
		/*return (0);*/ /* THIS SHOULD NOT BE FATAL! */
	}
	pw->pw_uid = id;

	if (!(p = strsep(&bp, ":")))			/* gid */
		goto fmt;
	if (p[0])
#ifdef __APPLE__
		/* nothing */;
#else /* !__APPLE__ */
		pw->pw_fields |= _PWF_GID;
#endif /* !__APPLE__ */
	else {
		if (pw->pw_name[0] != '+' && pw->pw_name[0] != '-') {
			if (flags & _PWSCAN_WARN)
				warnx("no gid for user %s", pw->pw_name);
			return (0);
		}
	}
	errno = 0;
#ifdef __APPLE__
	temp = strtol(p, &ep, 10);
	if ((temp == LONG_MAX && errno == ERANGE) || temp > GID_MAX) {
#else /* !__APPLE__ */
	temp = strtoul(p, &ep, 10);
	if ((temp == ULONG_MAX && errno == ERANGE) || temp > GID_MAX) {
#endif /* !__APPLE__ */
		if (flags & _PWSCAN_WARN)
			warnx("%s > max gid value (%u)", p, GID_MAX);
		return (0);
	}
#ifdef __APPLE__
	id = (id_t)temp;
#else /* !__APPLE__ */
	id = temp;
#endif /* !__APPLE__ */
	if (*ep != '\0') {
		if (flags & _PWSCAN_WARN)
			warnx("%s gid is incorrect", p);
		return (0);
	}
	if (flags & _PWSCAN_WARN && pw_big_ids_warning && id > USHRT_MAX) {
		warnx("%s > recommended max gid value (%u)", p, USHRT_MAX);
		/* return (0); This should not be fatal! */
	}
	pw->pw_gid = id;

	if (flags & _PWSCAN_MASTER ) {
		if (!(pw->pw_class = strsep(&bp, ":")))	/* class */
			goto fmt;
#ifndef __APPLE__
		if (pw->pw_class[0])
			pw->pw_fields |= _PWF_CLASS;
#endif /* !__APPLE__ */
		if (!(p = strsep(&bp, ":")))		/* change */
			goto fmt;
#ifndef __APPLE__
		if (p[0])
			pw->pw_fields |= _PWF_CHANGE;
#endif /* !__APPLE__ */
		pw->pw_change = atol(p);
		
		if (!(p = strsep(&bp, ":")))		/* expire */
			goto fmt;
#ifndef __APPLE__
		if (p[0])
			pw->pw_fields |= _PWF_EXPIRE;
#endif /* !__APPLE__ */
		pw->pw_expire = atol(p);
	}
	if (!(pw->pw_gecos = strsep(&bp, ":")))		/* gecos */
		goto fmt;
#ifndef __APPLE__
	if (pw->pw_gecos[0])
		pw->pw_fields |= _PWF_GECOS;
#endif /* !__APPLE__ */

	if (!(pw->pw_dir = strsep(&bp, ":")))		/* directory */
		goto fmt;
#ifndef __APPLE__
	if (pw->pw_dir[0])
		pw->pw_fields |= _PWF_DIR;
#endif /* !__APPLE__ */

	if (!(pw->pw_shell = strsep(&bp, ":")))		/* shell */
		goto fmt;

	p = pw->pw_shell;
	if (root && *p) {				/* empty == /bin/sh */
		for (setusershell();;) {
			if (!(sh = getusershell())) {
				if (flags & _PWSCAN_WARN)
					warnx("warning, unknown root shell");
				break;
			}
			if (!strcmp(p, sh))
				break;
		}
		endusershell();
	}
#ifndef __APPLE__
	if (p[0])
		pw->pw_fields |= _PWF_SHELL;
#endif /* !__APPLE__ */

	if ((p = strsep(&bp, ":"))) {			/* too many */
fmt:		
		if (flags & _PWSCAN_WARN)
			warnx("corrupted entry");
		return (0);
	}
	return (1);
}
