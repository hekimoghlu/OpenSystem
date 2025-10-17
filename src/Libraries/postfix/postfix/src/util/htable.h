/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 5, 2021.
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

#ifndef _HTABLE_H_INCLUDED_
#define _HTABLE_H_INCLUDED_

/*++
/* NAME
/*	htable 3h
/* SUMMARY
/*	hash table manager
/* SYNOPSIS
/*	#include <htable.h>
/* DESCRIPTION
/* .nf

 /* Structure of one hash table entry. */

typedef struct HTABLE_INFO {
    char   *key;			/* lookup key */
    void   *value;			/* associated value */
    struct HTABLE_INFO *next;		/* colliding entry */
    struct HTABLE_INFO *prev;		/* colliding entry */
} HTABLE_INFO;

 /* Structure of one hash table. */

typedef struct HTABLE {
    ssize_t size;			/* length of entries array */
    ssize_t used;			/* number of entries in table */
    HTABLE_INFO **data;			/* entries array, auto-resized */
    HTABLE_INFO **seq_bucket;		/* current sequence hash bucket */
    HTABLE_INFO **seq_element;		/* current sequence element */
} HTABLE;

extern HTABLE *htable_create(ssize_t);
extern HTABLE_INFO *htable_enter(HTABLE *, const char *, void *);
extern HTABLE_INFO *htable_locate(HTABLE *, const char *);
extern void *htable_find(HTABLE *, const char *);
extern void htable_delete(HTABLE *, const char *, void (*) (void *));
extern void htable_free(HTABLE *, void (*) (void *));
extern void htable_walk(HTABLE *, void (*) (HTABLE_INFO *, void *), void *);
extern HTABLE_INFO **htable_list(HTABLE *);
extern HTABLE_INFO *htable_sequence(HTABLE *, int);

#define HTABLE_SEQ_FIRST	0
#define HTABLE_SEQ_NEXT		1
#define HTABLE_SEQ_STOP		(-1)

 /*
  * Correct only when casting (char *) to (void *).
  */
#define HTABLE_ACTION_FN_CAST(f) ((void *)(HTABLE_INFO *, void *)) (f)
#define HTABLE_FREE_FN_CAST(f) ((void *)(void *)) (f)

/* LICENSE
/* .ad
/* .fi
/*	The Secure Mailer license must be distributed with this software.
/* AUTHOR(S)
/*	Wietse Venema
/*	IBM T.J. Watson Research
/*	P.O. Box 704
/*	Yorktown Heights, NY 10598, USA
/* CREATION DATE
/*	Fri Feb 14 13:43:19 EST 1997
/* LAST MODIFICATION
/*	%E% %U%
/* VERSION/RELEASE
/*	%I%
/*--*/

#endif
