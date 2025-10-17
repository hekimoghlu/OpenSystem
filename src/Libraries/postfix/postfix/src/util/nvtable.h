/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 31, 2022.
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

#ifndef _NVTABLE_H_INCLUDED_
#define _NVTABLE_H_INCLUDED_

/*++
/* NAME
/*	nvtable 3h
/* SUMMARY
/*	attribute list manager
/* SYNOPSIS
/*	#include <nvtable.h>
/* DESCRIPTION
/* .nf

 /*
  * Utility library.
  */
#include <htable.h>
#include <mymalloc.h>

typedef struct HTABLE NVTABLE;
typedef struct HTABLE_INFO NVTABLE_INFO;

#define nvtable_create(size)		htable_create(size)
#define nvtable_locate(table, key)	htable_locate((table), (key))
#define nvtable_walk(table, action, ptr) htable_walk((table), HTABLE_ACTION_FN_CAST(action), (ptr))
#define nvtable_list(table)		htable_list(table)
#define nvtable_find(table, key)	htable_find((table), (key))
#define nvtable_delete(table, key)	htable_delete((table), (key), myfree)
#define nvtable_free(table)		htable_free((table), myfree)

extern NVTABLE_INFO *nvtable_update(NVTABLE *, const char *, const char *);

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
