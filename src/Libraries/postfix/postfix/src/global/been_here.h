/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 30, 2025.
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

#ifndef _BEEN_HERE_H_INCLUDED_
#define _BEEN_HERE_H_INCLUDED_

/*++
/* NAME
/*	been_here 3h
/* SUMMARY
/*	detect repeated occurrence of string
/* SYNOPSIS
/*	#include <been_here.h>
/* DESCRIPTION
/* .nf

 /*
  * System library.
  */
#include <stdarg.h>

 /*
  * External interface.
  */
typedef struct {
    int     limit;			/* ceiling, zero for none */
    int     flags;			/* see below */
    struct HTABLE *table;
} BH_TABLE;

#define BH_FLAG_NONE	0		/* no special processing */
#define BH_FLAG_FOLD	(1<<0)		/* fold case */

extern BH_TABLE *been_here_init(int, int);
extern void been_here_free(BH_TABLE *);
extern int been_here_fixed(BH_TABLE *, const char *);
extern int PRINTFLIKE(2, 3) been_here(BH_TABLE *, const char *,...);
extern int been_here_check_fixed(BH_TABLE *, const char *);
extern int PRINTFLIKE(2, 3) been_here_check(BH_TABLE *, const char *,...);

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
