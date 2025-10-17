/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 12, 2024.
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

#ifndef _MALLOC_H_INCLUDED_
#define _MALLOC_H_INCLUDED_

/*++
/* NAME
/*	mymalloc 3h
/* SUMMARY
/*	memory management wrappers
/* SYNOPSIS
/*	#include "mymalloc.h"
/* DESCRIPTION
/* .nf

 /*
  * External interface.
  */
extern void *mymalloc(ssize_t);
extern void *myrealloc(void *, ssize_t);
extern void myfree(void *);
extern char *mystrdup(const char *);
extern char *mystrndup(const char *, ssize_t);
extern char *mymemdup(const void *, ssize_t);

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
