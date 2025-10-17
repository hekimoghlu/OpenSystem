/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 2, 2025.
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
#ifndef _TELLDIR_H_
#define	_TELLDIR_H_

#include <sys/queue.h>
#include <stdbool.h>

/*
 * One of these structures is malloced to describe the current directory
 * position each time telldir is called. It records the current magic
 * cookie returned by getdirentries and the offset within the buffer
 * associated with that return value.
 */
struct ddloc {
	LIST_ENTRY(ddloc) loc_lqe; /* entry in list */
	long	loc_index;	/* key associated with structure */
#if __DARWIN_64_BIT_INO_T
	__darwin_off_t	loc_seek;	/* returned by lseek */
#else /* !__DARWIN_64_BIT_INO_T */
	long	loc_seek;	/* magic cookie returned by getdirentries */
#endif /* __DARWIN_64_BIT_INO_T */
	long	loc_loc;	/* offset of entry in buffer */
};

/*
 * One of these structures is malloced for each DIR to record telldir
 * positions.
 */
struct _telldir {
	LIST_HEAD(, ddloc) td_locq; /* list of locations */
	long	td_loccnt;	/* index of entry for sequential readdir's */
#if __DARWIN_64_BIT_INO_T
	__darwin_off_t seekoff;	/* 64-bit seek offset */
#endif /* __DARWIN_64_BIT_INO_T */
};

/*
 * This lets paths like `/` or top-level bundles to return in a single
 * __getdirentries64 while keeping pressure on malloc small.
 */
#define READDIR_INITIAL_SIZE  2048
#define READDIR_LARGE_SIZE    (8 << 10)

#if __DARWIN_64_BIT_INO_T
size_t		__getdirentries64(int fd, void *buf, size_t bufsize, __darwin_off_t *basep);
#endif /* __DARWIN_64_BIT_INO_T */
__attribute__ ((visibility ("hidden")))
bool           _filldir(DIR *, bool) __DARWIN_INODE64(_filldir);
struct dirent	*_readdir_unlocked(DIR *, int) __DARWIN_INODE64(_readdir_unlocked);
void		_reclaim_telldir(DIR *);
void		_seekdir(DIR *, long) __DARWIN_ALIAS_I(_seekdir);
__attribute__ ((visibility ("hidden")))
void        _fixtelldir(DIR *dirp, long oldseek, long oldloc) __DARWIN_INODE64(_fixtelldir);
long		telldir(DIR *) __DARWIN_ALIAS_I(telldir);

#endif
