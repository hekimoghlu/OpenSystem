/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 2, 2022.
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

#ifndef _DICT_DB_H_INCLUDED_
#define _DICT_DB_H_INCLUDED_

/*++
/* NAME
/*	dict_db 3h
/* SUMMARY
/*	dictionary manager interface to DB files
/* SYNOPSIS
/*	#include <dict_db.h>
/* DESCRIPTION
/* .nf

 /*
  * Utility library.
  */
#include <dict.h>

 /*
  * External interface.
  */
#define DICT_TYPE_HASH	"hash"
#define DICT_TYPE_BTREE	"btree"

extern DICT *dict_hash_open(const char *, int, int);
extern DICT *dict_btree_open(const char *, int, int);

 /*
  * XXX Should be part of the DICT interface.
  * 
  * You can override the default dict_db_cache_size setting before calling
  * dict_hash_open() or dict_btree_open(). This is done in mkmap_db_open() to
  * set a larger memory pool for database (re)builds.
  */
extern int dict_db_cache_size;

#define DEFINE_DICT_DB_CACHE_SIZE int dict_db_cache_size = (128 * 1024)

/* LICENSE
/* .ad
/* .fi
/*	The Secure Mailer license must be distributed with this software.
/* AUTHOR(S)
/*	Wietse Venema
/*	IBM T.J. Watson Research
/*	P.O. Box 704
/*	Yorktown Heights, NY 10598, USA
/*--*/

#endif
