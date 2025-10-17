/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 13, 2025.
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
#if 0
#ifndef lint
static char sccsid[] = "@(#)misc.c	8.1 (Berkeley) 6/6/93";
#endif /*not lint */
#endif
#include <sys/cdefs.h>
#include <errno.h>
__FBSDID("$FreeBSD: src/usr.sbin/mtree/misc.c,v 1.16 2005/03/29 11:44:17 tobez Exp $");

#include <sys/types.h>
#include <sys/stat.h>
#include <err.h>
#include <fts.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#include "metrics.h"
#include "mtree.h"
#include "extern.h"
#import <sys/attr.h>
#include <vis.h>

typedef struct _key {
	const char *name;			/* key name */
	u_int64_t val;				/* value */

#define	NEEDVALUE	0x01
	u_int flags;
} KEY;

/* NB: the following table must be sorted lexically. */
static KEY keylist[] = {
	{"acldigest",			F_ACL,			NEEDVALUE},
	{"atime",			F_ATIME,		NEEDVALUE},
	{"btime",			F_BTIME,		NEEDVALUE},
	{"cksum",			F_CKSUM,		NEEDVALUE},
	{"ctime",			F_CTIME,		NEEDVALUE},
	{"dataless",			F_DATALESS,		NEEDVALUE},
	{"flags",			F_FLAGS,		NEEDVALUE},
	{"gid",				F_GID,			NEEDVALUE},
	{"gname",			F_GNAME,		NEEDVALUE},
	{"ignore",			F_IGN,			0},
	{"inode",			F_INODE,		NEEDVALUE},
	{"link",			F_SLINK,		NEEDVALUE},
#ifdef ENABLE_MD5
	{"md5digest",			F_MD5,			NEEDVALUE},
#endif
	{"mode",			F_MODE,			NEEDVALUE},
	{"nlink",			F_NLINK,		NEEDVALUE},
	{"nochange",			F_NOCHANGE,		0},
	{"nxattr",			F_NXATTR,		NEEDVALUE},
	{"protectionclass",		F_PROTECTION_CLASS,	NEEDVALUE},
	{"ptime",			F_PTIME,		NEEDVALUE},
	{"purgeable",			F_PURGEABLE,		NEEDVALUE},
#ifdef ENABLE_RMD160
	{"ripemd160digest",		F_RMD160,		NEEDVALUE},
#endif
#ifdef ENABLE_SHA1
	{"sha1digest",			F_SHA1,			NEEDVALUE},
#endif
#ifdef ENABLE_SHA256
	{"sha256digest",		F_SHA256,		NEEDVALUE},
#endif
	{"siblingid",			F_SIBLINGID,		NEEDVALUE},
	{"size",			F_SIZE,			NEEDVALUE},
	{"time",			F_TIME,			NEEDVALUE},
	{"type",			F_TYPE,			NEEDVALUE},
	{"uid",				F_UID,			NEEDVALUE},
	{"uname",			F_UNAME,		NEEDVALUE},
	{"xattrsdigest",		F_XATTRS,		NEEDVALUE},
};

int keycompare(const void *, const void *);

u_int64_t
parsekey(char *name, int *needvaluep)
{
	KEY *k, tmp;

	tmp.name = name;
	k = (KEY *)bsearch(&tmp, keylist, sizeof(keylist) / sizeof(KEY),
	    sizeof(KEY), keycompare);
	if (k == NULL) {
		RECORD_FAILURE(107, EINVAL);
		errx(1, "line %d: unknown keyword %s", lineno, name);
	}

	if (needvaluep)
		*needvaluep = k->flags & NEEDVALUE ? 1 : 0;
	return (k->val);
}

int
keycompare(const void *a, const void *b)
{
	return (strcmp(((const KEY *)a)->name, ((const KEY *)b)->name));
}

char *
flags_to_string(u_long fflags)
{
	int error = 0;
	char *string;

	string = fflagstostr(fflags);
	if (string != NULL && *string == '\0') {
		free(string);
		string = strdup("none");
	}
	if (string == NULL) {
		error = errno;
		RECORD_FAILURE(108, error);
		errc(1, error, NULL);
	}

	return string;
}

// escape path and always return a new string so it can be freed
char *
escape_path(char *string)
{
	char *escapedPath = calloc(1, strlen(string) * 4  +  1);
	if (escapedPath == NULL) {
		RECORD_FAILURE(109, ENOMEM);
		errx(1, "escape_path(): calloc() failed");
	}
	strvis(escapedPath, string, VIS_NL | VIS_CSTYLE | VIS_OCTAL);
	
	return escapedPath;
}

struct ptimebuf {
	uint32_t length;
	attribute_set_t returned_attrs;
	struct timespec st_ptimespec;
} __attribute__((aligned(4), packed));

// ptime is not supported on root filesystems or HFS filesystems older than the feature being introduced
struct timespec
ptime(char *path, int *supported) {
	
	int error = 0;
	int ret = 0;
	struct ptimebuf buf;
	struct attrlist list = {
		.bitmapcount = ATTR_BIT_MAP_COUNT,
		.commonattr = ATTR_CMN_RETURNED_ATTRS | ATTR_CMN_ADDEDTIME,
	};
	ret = getattrlist(path, &list, &buf, sizeof(buf), FSOPT_NOFOLLOW);
	if (ret) {
		error = errno;
		RECORD_FAILURE(110, error);
		errc(1, error, "ptime: getattrlist");
	}
	
	*supported = 0;
	if (buf.returned_attrs.commonattr & ATTR_CMN_ADDEDTIME) {
		*supported = 1;
	}
	
	return buf.st_ptimespec;
	
}
