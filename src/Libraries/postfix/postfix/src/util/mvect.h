/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 28, 2023.
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

#ifndef _MVECT_H_INCLUDED_
#define _MVECT_H_INCLUDED_

/*++
/* NAME
/*	mvect 3h
/* SUMMARY
/*	memory vector management
/* SYNOPSIS
/*	#include <mvect.h>
/* DESCRIPTION
/* .nf

 /*
  * Generic memory vector interface.
  */
typedef void (*MVECT_FN) (char *, ssize_t);

typedef struct {
    char   *ptr;
    ssize_t elsize;
    ssize_t nelm;
    MVECT_FN init_fn;
    MVECT_FN wipe_fn;
} MVECT;

extern char *mvect_alloc(MVECT *, ssize_t, ssize_t, MVECT_FN, MVECT_FN);
extern char *mvect_realloc(MVECT *, ssize_t);
extern char *mvect_free(MVECT *);

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
