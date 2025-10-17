/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 16, 2025.
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
#ifndef _MPOOL_H_
#define _MPOOL_H_

#include <sys/queue.h>

/*
 * The memory pool scheme is a simple one.  Each in-memory page is referenced
 * by a bucket which is threaded in up to two of three ways.  All active pages
 * are threaded on a hash chain (hashed by page number) and an lru chain.
 * Inactive pages are threaded on a free chain.  Each reference to a memory
 * pool is handed an opaque MPOOL cookie which stores all of this information.
 */
#define	HASHSIZE	128
#define	HASHKEY(pgno)	((pgno - 1) % HASHSIZE)

/* The BKT structures are the elements of the queues. */
typedef struct _bkt {
	TAILQ_ENTRY(_bkt) hq;		/* hash queue */
	TAILQ_ENTRY(_bkt) q;		/* lru queue */
	void    *page;			/* page */
	pgno_t   pgno;			/* page number */

#define	MPOOL_DIRTY	0x01		/* page needs to be written */
#define	MPOOL_PINNED	0x02		/* page is pinned into memory */
	u_int8_t flags;			/* flags */
} BKT;

typedef struct MPOOL {
	TAILQ_HEAD(_lqh, _bkt) lqh;	/* lru queue head */
					/* hash queue array */
	TAILQ_HEAD(_hqh, _bkt) hqh[HASHSIZE];
	pgno_t	curcache;		/* current number of cached pages */
	pgno_t	maxcache;		/* max number of cached pages */
	pgno_t	npages;			/* number of pages in the file */
	unsigned long	pagesize;	/* file page size */
	int	fd;			/* file descriptor */
					/* page in conversion routine */
	void    (*pgin)(void *, pgno_t, void *);
					/* page out conversion routine */
	void    (*pgout)(void *, pgno_t, void *);
	void	*pgcookie;		/* cookie for page in/out routines */
#ifdef STATISTICS
	unsigned long	cachehit;
	unsigned long	cachemiss;
	unsigned long	pagealloc;
	unsigned long	pageflush;
	unsigned long	pageget;
	unsigned long	pagenew;
	unsigned long	pageput;
	unsigned long	pageread;
	unsigned long	pagewrite;
#endif
} MPOOL;

__BEGIN_DECLS
MPOOL	*mpool_open(void *, int, pgno_t, pgno_t);
void	 mpool_filter(MPOOL *, void (*)(void *, pgno_t, void *),
	    void (*)(void *, pgno_t, void *), void *);
void	*mpool_new(MPOOL *, pgno_t *);
void	*mpool_get(MPOOL *, pgno_t, unsigned int);
int	 mpool_put(MPOOL *, void *, unsigned int);
int	 mpool_sync(MPOOL *);
int	 mpool_close(MPOOL *);
#ifdef STATISTICS
void	 mpool_stat(MPOOL *);
#endif
__END_DECLS

#endif
