/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 13, 2023.
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
#ifndef	_SDBM_H_
#define	_SDBM_H_

#include <stdio.h>

#define DBLKSIZ 4096
#define PBLKSIZ 1024
#define PAIRMAX 1008			/* arbitrary on PBLKSIZ-N */
#define SPLTMAX	10			/* maximum allowed splits */
					/* for a single insertion */
#define DIRFEXT	".dir"
#define PAGFEXT	".pag"

typedef struct {
	int dirf;		       /* directory file descriptor */
	int pagf;		       /* page file descriptor */
	int flags;		       /* status/error flags, see below */
	int keyptr;		       /* current key for nextkey */
	off_t maxbno;		       /* size of dirfile in bits */
	long curbit;		       /* current bit number */
	long hmask;		       /* current hash mask */
	long blkptr;		       /* current block for nextkey */
	long blkno;		       /* current page to read/write */
	long pagbno;		       /* current page in pagbuf */
	char pagbuf[PBLKSIZ];	       /* page file block buffer */
	long dirbno;		       /* current block in dirbuf */
	char dirbuf[DBLKSIZ];	       /* directory file block buffer */
} DBM;

#define DBM_RDONLY	0x1	       /* data base open read-only */
#define DBM_IOERR	0x2	       /* data base I/O error */

/*
 * utility macros
 */
#define sdbm_rdonly(db)		((db)->flags & DBM_RDONLY)
#define sdbm_error(db)		((db)->flags & DBM_IOERR)

#define sdbm_clearerr(db)	((db)->flags &= ~DBM_IOERR)  /* ouch */

#define sdbm_dirfno(db)	((db)->dirf)
#define sdbm_pagfno(db)	((db)->pagf)

typedef struct {
	char *dptr;
	int dsize;
} datum;

extern datum nullitem;

#if defined(__STDC__)
#define proto(p) p
#else
#define proto(p) ()
#endif

/*
 * flags to sdbm_store
 */
#define DBM_INSERT	0
#define DBM_REPLACE	1

/*
 * ndbm interface
 */
extern DBM *sdbm_open proto((char *, int, int));
extern void sdbm_close proto((DBM *));
extern datum sdbm_fetch proto((DBM *, datum));
extern int sdbm_delete proto((DBM *, datum));
extern int sdbm_store proto((DBM *, datum, datum, int));
extern datum sdbm_firstkey proto((DBM *));
extern datum sdbm_nextkey proto((DBM *));

/*
 * other
 */
extern DBM *sdbm_prep proto((char *, char *, int, int));
extern long sdbm_hash proto((char *, int));

#endif	/* _SDBM_H_ */
