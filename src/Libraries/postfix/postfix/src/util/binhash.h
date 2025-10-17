/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 8, 2025.
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

#ifndef _BINHASH_H_INCLUDED_
#define _BINHASH_H_INCLUDED_

/*++
/* NAME
/*	binhash 3h
/* SUMMARY
/*	hash table manager
/* SYNOPSIS
/*	#include <binhash.h>
/* DESCRIPTION
/* .nf

 /* Structure of one hash table entry. */

typedef struct BINHASH_INFO {
    void   *key;			/* lookup key */
    ssize_t key_len;			/* key length */
    void   *value;			/* associated value */
    struct BINHASH_INFO *next;		/* colliding entry */
    struct BINHASH_INFO *prev;		/* colliding entry */
} BINHASH_INFO;

 /* Structure of one hash table. */

typedef struct BINHASH {
    ssize_t size;			/* length of entries array */
    ssize_t used;			/* number of entries in table */
    BINHASH_INFO **data;		/* entries array, auto-resized */
} BINHASH;

extern BINHASH *binhash_create(ssize_t);
extern BINHASH_INFO *binhash_enter(BINHASH *, const void *, ssize_t, void *);
extern BINHASH_INFO *binhash_locate(BINHASH *, const void *, ssize_t);
extern void *binhash_find(BINHASH *, const void *, ssize_t);
extern void binhash_delete(BINHASH *, const void *, ssize_t, void (*) (void *));
extern void binhash_free(BINHASH *, void (*) (void *));
extern void binhash_walk(BINHASH *, void (*) (BINHASH_INFO *, void *), void *);
extern BINHASH_INFO **binhash_list(BINHASH *);

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
/*	Thu Feb 20 16:54:29 EST 1997
/* LAST MODIFICATION
/*	%E% %U%
/* VERSION/RELEASE
/*	%I%
/*--*/

#endif
