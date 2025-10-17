/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 14, 2024.
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
#ifndef _CACHE_H
#define _CACHE_H


/* constant includes */
#include "saslauthd.h"


/****************************************************************
* * Plug in some autoconf magic to determine what implementation
* * to use for the table slot (row) locking.
****************************************************************/
#ifdef USE_DOORS
# define CACHE_USE_PTHREAD_RWLOCK
#else
# define CACHE_USE_FCNTL
#endif



/************************************************/
#ifdef CACHE_USE_FCNTL
	/* FCNTL Impl */

struct lock_ctl {
	char			*flock_file;
	int			flock_fd;
};

#endif  /* CACHE_USE_FCNTL */
/************************************************/



/************************************************/
#ifdef CACHE_USE_PTHREAD_RWLOCK
	/* RWLock Impl */

#include <pthread.h>

struct lock_ctl {
	pthread_rwlock_t	*rwlock;
};

#endif  /* CACHE_USE_PTHREAD_RWLOCK */
/************************************************/



/* defaults */
#define CACHE_DEFAULT_TIMEOUT		28800
#define CACHE_DEFAULT_TABLE_SIZE	1711
#define CACHE_DEFAULT_FLAGS		0
#define CACHE_MAX_BUCKETS_PER		6
#define CACHE_MMAP_FILE			"/cache.mmap"  /* don't forget the "/" */
#define CACHE_FLOCK_FILE		"/cache.flock" /* don't forget the "/" */



/* If debugging uncomment this for always verbose  */
/* #define CACHE_DEFAULT_FLAGS		CACHE_VERBOSE */



/* max length for cached credential values */
#define CACHE_MAX_CREDS_LENGTH		60



/* magic values (must be less than 63 chars!) */
#define CACHE_CACHE_MAGIC		"SASLAUTHD_CACHE_MAGIC"



/* return values */
#define CACHE_OK			0
#define CACHE_FAIL			1
#define CACHE_TOO_BIG			2	



/* cache_result status values */
#define CACHE_NO_FLUSH			0
#define CACHE_FLUSH			1
#define CACHE_FLUSH_WITH_RESCAN		2	



/* declarations */
struct bucket {
        char            	creds[CACHE_MAX_CREDS_LENGTH];
        unsigned int		user_offt;
        unsigned int		realm_offt;
        unsigned int		service_offt;
        unsigned char   	pwd_digest[16];
        time_t          	created;
};

struct stats {
        volatile unsigned int   hits;
        volatile unsigned int   misses;
        volatile unsigned int   lock_failures;
        volatile unsigned int   attempts;
        unsigned int            table_size;
        unsigned int            max_buckets_per;
        unsigned int            sizeof_bucket;
        unsigned int            bytes;
        unsigned int            timeout;
};

struct mm_ctl {
	void			*base;
	unsigned int		bytes;
	char			*file;
};

struct cache_result {
	struct bucket		bucket;
	struct bucket   	*read_bucket;
	unsigned int		hash_offset;
	int			status;
};


/* cache.c */
extern int cache_init(void);
extern int cache_lookup(const char *, const char *, const char *, const char *, struct cache_result *);
extern void cache_commit(struct cache_result *);
extern int cache_pjwhash(char *);
extern void cache_set_table_size(const char *);
extern void cache_set_timeout(const char *);
extern unsigned int cache_get_next_prime(unsigned int);
extern void *cache_alloc_mm(unsigned int);
extern void cache_cleanup_mm(void);
extern void cache_cleanup_lock(void);
extern int cache_init_lock(void);
extern int cache_get_wlock(unsigned int);
extern int cache_get_rlock(unsigned int);
extern int cache_un_lock(unsigned int);

#endif  /* _CACHE_H */

