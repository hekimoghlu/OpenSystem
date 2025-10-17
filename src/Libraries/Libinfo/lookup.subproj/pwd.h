/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 7, 2022.
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
/* Portions copyright (c) 2000-2011 Apple Inc. All rights reserved. */

#ifndef _PWD_H_
#define	_PWD_H_

#include <_types.h>
#include <sys/_types/_gid_t.h>
#include <sys/_types/_size_t.h>
#include <sys/_types/_uid_t.h>

#if !defined(_POSIX_C_SOURCE) || defined(_DARWIN_C_SOURCE)
#define	_PATH_PWD		"/etc"
#define	_PATH_PASSWD		"/etc/passwd"
#define	_PASSWD			"passwd"
#define	_PATH_MASTERPASSWD	"/etc/master.passwd"
#define	_PATH_MASTERPASSWD_LOCK	"/etc/ptmp"
#define	_MASTERPASSWD		"master.passwd"

#define	_PATH_MP_DB		"/etc/pwd.db"
#define	_MP_DB			"pwd.db"
#define	_PATH_SMP_DB		"/etc/spwd.db"
#define	_SMP_DB			"spwd.db"

#define	_PATH_PWD_MKDB		"/usr/sbin/pwd_mkdb"

#define	_PW_KEYBYNAME		'1'	/* stored by name */
#define	_PW_KEYBYNUM		'2'	/* stored by entry in the "file" */
#define	_PW_KEYBYUID		'3'	/* stored by uid */

#define	_PASSWORD_EFMT1		'_'	/* extended encryption format */

#define	_PASSWORD_LEN		128	/* max length, not counting NULL */

#define _PASSWORD_NOUID		0x01	/* flag for no specified uid. */
#define _PASSWORD_NOGID		0x02	/* flag for no specified gid. */
#define _PASSWORD_NOCHG		0x04	/* flag for no specified change. */
#define _PASSWORD_NOEXP		0x08	/* flag for no specified expire. */

#define _PASSWORD_WARNDAYS	14	/* days to warn about expiry */
#define _PASSWORD_CHGNOW	-1	/* special day to force password
					 * change at next login */
#endif

struct passwd {
	char	*pw_name;		/* user name */
	char	*pw_passwd;		/* encrypted password */
	uid_t	pw_uid;			/* user uid */
	gid_t	pw_gid;			/* user gid */
	__darwin_time_t pw_change;		/* password change time */
	char	*pw_class;		/* user access class */
	char	*pw_gecos;		/* Honeywell login info */
	char	*pw_dir;		/* home directory */
	char	*pw_shell;		/* default shell */
	__darwin_time_t pw_expire;		/* account expiration */
};

#include <sys/cdefs.h>

__BEGIN_DECLS
struct passwd	*getpwuid(uid_t);
struct passwd	*getpwnam(const char *);
int		 getpwuid_r(uid_t, struct passwd *, char *, size_t, struct passwd **);
int		 getpwnam_r(const char *, struct passwd *, char *, size_t, struct passwd **);
struct passwd	*getpwent(void);
void		 setpwent(void);
void		 endpwent(void);
__END_DECLS

#if (!defined(_POSIX_C_SOURCE) && !defined(_XOPEN_SOURCE)) || defined(_DARWIN_C_SOURCE)
#include <uuid/uuid.h>
__BEGIN_DECLS
int		 setpassent(int);
char 		*user_from_uid(uid_t, int);
struct passwd	*getpwuuid(uuid_t);
int		 getpwuuid_r(uuid_t, struct passwd *, char *, size_t, struct passwd **);
__END_DECLS
#endif

#endif /* !_PWD_H_ */
