/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 14, 2025.
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
/*	$NetBSD: mntopts.h,v 1.7 2006/02/12 01:32:06 chs Exp $	*/

/*-
 * Copyright (c) 1994
 *      The Regents of the University of California.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the University nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 *
 *	@(#)mntopts.h	8.7 (Berkeley) 3/29/95
 */
#ifndef _MNTOPTS_H_
#define _MNTOPTS_H_

#ifdef __cplusplus
extern "C" {
#endif

struct mntopt {
	const char *m_option;	/* option name */
	int m_inverse;		/* if a negative option, eg "dev" */
	int m_flag;		/* bit to set, eg. MNT_RDONLY */
	int m_altloc;		/* 1 => set bit in altflags */
};

/* User-visible MNT_ flags. */
#define MOPT_ASYNC		{ "async",	0, MNT_ASYNC, 0 }
#define MOPT_NODEV		{ "dev",	1, MNT_NODEV, 0 }
#define MOPT_NOEXEC		{ "exec",	1, MNT_NOEXEC, 0 }
#define MOPT_NOSUID		{ "suid",	1, MNT_NOSUID, 0 }
#define MOPT_RDONLY		{ "rdonly",	0, MNT_RDONLY, 0 }
#define MOPT_SYNC		{ "sync",	0, MNT_SYNCHRONOUS, 0 }
#define MOPT_UNION		{ "union",	0, MNT_UNION, 0 }
#define MOPT_USERQUOTA		{ "userquota",	0, 0, 0 }
#define MOPT_GROUPQUOTA		{ "groupquota",	0, 0, 0 }
#define MOPT_BROWSE		{ "browse",	1, MNT_DONTBROWSE, 0 }
#define MOPT_AUTOMOUNTED	{ "automounted",0, MNT_AUTOMOUNTED, 0 }
#define MOPT_DEFWRITE		{ "defwrite",	0, MNT_DEFWRITE, 0}
#define	MOPT_NOATIME		{ "atime",	1, MNT_NOATIME, 0}
#define MOPT_NOFOLLOW		{ "follow", 1, MNT_NOFOLLOW, 0}
#define MOPT_IGNORE_OWNERSHIP	{ "owners",	1, MNT_IGNORE_OWNERSHIP, 0}
/* alias the deprecated name for compatibility */
#define MOPT_PERMISSIONS	{ "perm",	1, MNT_IGNORE_OWNERSHIP, 0}
#define	MOPT_QUARANTINE		{ "quarantine",	0, MNT_QUARANTINE, 0}
#define MOPT_CPROTECT		{ "protect",	0, MNT_CPROTECT, 0 }

/* Control flags. */
#define MOPT_FORCE		{ "force",	0, MNT_FORCE, 0 }
#define MOPT_UPDATE		{ "update",	0, MNT_UPDATE, 0 }
#define MOPT_RELOAD		{ "reload",	0, MNT_RELOAD, 0 }

/* Support for old-style "ro", "rw" flags. */
#define MOPT_RO			{ "ro",		0, MNT_RDONLY, 0 }
#define MOPT_RW			{ "rw",		1, MNT_RDONLY, 0 }

/* This is parsed by mount(8), but is ignored by specific mount_*(8)s. */
#define MOPT_AUTO		{ "auto",	0, 0, 0 }

#define MOPT_FSTAB_COMPAT						\
	MOPT_RO,							\
	MOPT_RW,							\
	MOPT_AUTO

/* Standard options which all mounts can understand. */
#define MOPT_STDOPTS							\
	MOPT_USERQUOTA,							\
	MOPT_GROUPQUOTA,						\
	MOPT_FSTAB_COMPAT,						\
	MOPT_NODEV,							\
	MOPT_NOEXEC,							\
	MOPT_NOSUID,							\
	MOPT_NOFOLLOW,							\
	MOPT_RDONLY,							\
	MOPT_UNION,							\
        MOPT_BROWSE,							\
        MOPT_AUTOMOUNTED,						\
        MOPT_DEFWRITE,							\
	MOPT_NOATIME,							\
	MOPT_PERMISSIONS,						\
	MOPT_IGNORE_OWNERSHIP,						\
	MOPT_QUARANTINE,							\
	MOPT_CPROTECT

typedef struct mntoptparse *mntoptparse_t;
mntoptparse_t getmntopts(const char *, const struct mntopt *, int *, int *);
const char *getmntoptstr(mntoptparse_t, const char *);
long getmntoptnum(mntoptparse_t, const char *);
void freemntopts(mntoptparse_t);

extern int getmnt_silent;

#ifdef __cplusplus
}
#endif

#endif /* _MNTOPTS_H_ */
