/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 4, 2024.
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
static char sccsid[] = "@(#)cmp.c	8.1 (Berkeley) 5/31/93";
#endif /* not lint */
#endif
#include <sys/cdefs.h>
__FBSDID("$FreeBSD$");


#include <sys/types.h>
#include <sys/stat.h>

#include <fts.h>
#include <string.h>

#include "ls.h"
#include "extern.h"

#ifdef __APPLE__
/*
 * Aliased to the ino64, non-POSIX-conformant names; this works with the default
 * build configuration of ls(1) on MacOS.
 */
#define	st_atim	st_atimespec
#define	st_mtim	st_mtimespec
#define	st_ctim	st_ctimespec
#define	st_birthtim	st_birthtimespec
#endif

int
namecmp(const FTSENT *a, const FTSENT *b)
{

	return (strcoll(a->fts_name, b->fts_name));
}

int
revnamecmp(const FTSENT *a, const FTSENT *b)
{

	return (strcoll(b->fts_name, a->fts_name));
}

int
modcmp(const FTSENT *a, const FTSENT *b)
{

	if (b->fts_statp->st_mtim.tv_sec >
	    a->fts_statp->st_mtim.tv_sec)
		return (1);
	if (b->fts_statp->st_mtim.tv_sec <
	    a->fts_statp->st_mtim.tv_sec)
		return (-1);
	if (b->fts_statp->st_mtim.tv_nsec >
	    a->fts_statp->st_mtim.tv_nsec)
		return (1);
	if (b->fts_statp->st_mtim.tv_nsec <
	    a->fts_statp->st_mtim.tv_nsec)
		return (-1);
	if (f_samesort)
		return (strcoll(b->fts_name, a->fts_name));
	else
		return (strcoll(a->fts_name, b->fts_name));
}

int
revmodcmp(const FTSENT *a, const FTSENT *b)
{

	return (modcmp(b, a));
}

int
acccmp(const FTSENT *a, const FTSENT *b)
{

	if (b->fts_statp->st_atim.tv_sec >
	    a->fts_statp->st_atim.tv_sec)
		return (1);
	if (b->fts_statp->st_atim.tv_sec <
	    a->fts_statp->st_atim.tv_sec)
		return (-1);
	if (b->fts_statp->st_atim.tv_nsec >
	    a->fts_statp->st_atim.tv_nsec)
		return (1);
	if (b->fts_statp->st_atim.tv_nsec <
	    a->fts_statp->st_atim.tv_nsec)
		return (-1);
	if (f_samesort)
		return (strcoll(b->fts_name, a->fts_name));
	else
		return (strcoll(a->fts_name, b->fts_name));
}

int
revacccmp(const FTSENT *a, const FTSENT *b)
{

	return (acccmp(b, a));
}

int
birthcmp(const FTSENT *a, const FTSENT *b)
{

	if (b->fts_statp->st_birthtim.tv_sec >
	    a->fts_statp->st_birthtim.tv_sec)
		return (1);
	if (b->fts_statp->st_birthtim.tv_sec <
	    a->fts_statp->st_birthtim.tv_sec)
		return (-1);
	if (b->fts_statp->st_birthtim.tv_nsec >
	    a->fts_statp->st_birthtim.tv_nsec)
		return (1);
	if (b->fts_statp->st_birthtim.tv_nsec <
	    a->fts_statp->st_birthtim.tv_nsec)
		return (-1);
	if (f_samesort)
		return (strcoll(b->fts_name, a->fts_name));
	else
		return (strcoll(a->fts_name, b->fts_name));
}

int
revbirthcmp(const FTSENT *a, const FTSENT *b)
{

	return (birthcmp(b, a));
}

int
statcmp(const FTSENT *a, const FTSENT *b)
{

	if (b->fts_statp->st_ctim.tv_sec >
	    a->fts_statp->st_ctim.tv_sec)
		return (1);
	if (b->fts_statp->st_ctim.tv_sec <
	    a->fts_statp->st_ctim.tv_sec)
		return (-1);
	if (b->fts_statp->st_ctim.tv_nsec >
	    a->fts_statp->st_ctim.tv_nsec)
		return (1);
	if (b->fts_statp->st_ctim.tv_nsec <
	    a->fts_statp->st_ctim.tv_nsec)
		return (-1);
	if (f_samesort)
		return (strcoll(b->fts_name, a->fts_name));
	else
		return (strcoll(a->fts_name, b->fts_name));
}

int
revstatcmp(const FTSENT *a, const FTSENT *b)
{

	return (statcmp(b, a));
}

int
sizecmp(const FTSENT *a, const FTSENT *b)
{

	if (b->fts_statp->st_size > a->fts_statp->st_size)
		return (1);
	if (b->fts_statp->st_size < a->fts_statp->st_size)
		return (-1);
	return (strcoll(a->fts_name, b->fts_name));
}

int
revsizecmp(const FTSENT *a, const FTSENT *b)
{

	return (sizecmp(b, a));
}
