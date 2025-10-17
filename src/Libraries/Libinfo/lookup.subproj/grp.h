/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 1, 2024.
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
/* Portions copyright (c) 2000-2018 Apple Inc. All rights reserved. */ 

#ifndef _GRP_H_
#define	_GRP_H_

#include <_types.h>
#include <sys/_types/_gid_t.h>	/* [XBD] */
#include <sys/_types/_size_t.h> /* SUSv4 */

#if !defined(_POSIX_C_SOURCE) || defined(_DARWIN_C_SOURCE)
#define	_PATH_GROUP		"/etc/group"
#endif

struct group {
	char	*gr_name;		/* [XBD] group name */
	char	*gr_passwd;		/* [???] group password */
	gid_t	gr_gid;			/* [XBD] group id */
	char	**gr_mem;		/* [XBD] group members */
};

#include <sys/cdefs.h>

__BEGIN_DECLS
/* [XBD] */
struct group *getgrgid(gid_t);
struct group *getgrnam(const char *);
/* [TSF] */
int getgrgid_r(gid_t, struct group *, char *, size_t, struct group **);
int getgrnam_r(const char *, struct group *, char *, size_t, struct group **);
/* [XSI] */
struct group *getgrent(void);
void setgrent(void);
void endgrent(void);
__END_DECLS

#if (!defined(_POSIX_C_SOURCE) && !defined(_XOPEN_SOURCE)) || defined(_DARWIN_C_SOURCE)
#include <uuid/uuid.h>
__BEGIN_DECLS
char *group_from_gid(gid_t, int);
struct group *getgruuid(uuid_t);
int getgruuid_r(uuid_t, struct group *, char *, size_t, struct group **);
__END_DECLS
#endif

#if !defined(_XOPEN_SOURCE) || defined(_DARWIN_C_SOURCE)
__BEGIN_DECLS
#if (!defined(LIBINFO_INSTALL_API) || !LIBINFO_INSTALL_API)
void setgrfile(const char *);
#endif
int setgroupent(int);
__END_DECLS
#endif

#endif /* !_GRP_H_ */
