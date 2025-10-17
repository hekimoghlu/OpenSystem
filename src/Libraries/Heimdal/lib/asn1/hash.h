/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 4, 2022.
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
/*
 * hash.h. Header file for hash table functions
 */

/* $Id$ */

struct hashentry {		/* Entry in bucket */
     struct hashentry **prev;
     struct hashentry *next;
     void *ptr;
};

typedef struct hashentry Hashentry;

struct hashtab {		/* Hash table */
     int (*cmp)(void *, void *); /* Compare function */
     unsigned (*hash)(void *);	/* hash function */
     int sz;			/* Size */
     Hashentry *tab[1];		/* The table */
};

typedef struct hashtab Hashtab;

/* prototypes */

Hashtab *hashtabnew(int sz,
		    int (*cmp)(void *, void *),
		    unsigned (*hash)(void *));	/* Make new hash table */

void *hashtabsearch(Hashtab *htab, /* The hash table */
		    void *ptr);	/*  The key */


void *hashtabadd(Hashtab *htab,	/* The hash table */
	       void *ptr);	/* The element */

int _hashtabdel(Hashtab *htab,	/* The table */
		void *ptr,	/* Key */
		int freep);	/* Free data part? */

void hashtabforeach(Hashtab *htab,
		    int (*func)(void *ptr, void *arg),
		    void *arg);

unsigned hashadd(const char *s);		/* Standard hash function */
unsigned hashcaseadd(const char *s);		/* Standard hash function */
unsigned hashjpw(const char *s);		/* another hash function */

/* macros */

 /* Don't free space */
#define hashtabdel(htab,key)  _hashtabdel(htab,key,FALSE)

#define hashtabfree(htab,key) _hashtabdel(htab,key,TRUE) /* Do! */
