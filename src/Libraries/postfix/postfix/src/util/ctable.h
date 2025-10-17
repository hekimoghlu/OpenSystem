/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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

#ifndef _CTABLE_H_INCLUDED_
#define _CTABLE_H_INCLUDED_

/*++
/* NAME
/*	ctable 5
/* SUMMARY
/*	cache manager
/* SYNOPSIS
/*	#include <ctable.h>
/* DESCRIPTION
/* .nf

 /*
  * Interface of the cache manager. The structure of a cache is not visible
  * to the caller.
  */

#define	CTABLE struct ctable
typedef void *(*CTABLE_CREATE_FN) (const char *, void *);
typedef void (*CTABLE_DELETE_FN) (void *, void *);

extern CTABLE *ctable_create(ssize_t, CTABLE_CREATE_FN, CTABLE_DELETE_FN, void *);
extern void ctable_free(CTABLE *);
extern void ctable_walk(CTABLE *, void (*) (const char *, const void *));
extern const void *ctable_locate(CTABLE *, const char *);
extern const void *ctable_refresh(CTABLE *, const char *);
extern void ctable_newcontext(CTABLE *, void *);

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
