/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 31, 2022.
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

#ifndef _DEFER_H_INCLUDED_
#define _DEFER_H_INCLUDED_

/*++
/* NAME
/*	defer 3h
/* SUMMARY
/*	defer service client interface
/* SYNOPSIS
/*	#include <defer.h>
/* DESCRIPTION
/* .nf

 /*
  * Global library.
  */
#include <bounce.h>

 /*
  * External interface.
  */
extern int defer_append(int, const char *, MSG_STATS *, RECIPIENT *,
			        const char *, DSN *);
extern int defer_flush(int, const char *, const char *, const char *, int,
		               const char *, const char *, int);
extern int defer_warn(int, const char *, const char *, const char *, int,
		              const char *, const char *, int);
extern int defer_one(int, const char *, const char *, const char *, int,
		             const char *, const char *,
		             int, MSG_STATS *, RECIPIENT *,
		             const char *, DSN *);

 /*
  * Start of private API.
  */
#ifdef DSN_INTERN

extern int defer_append_intern(int, const char *, MSG_STATS *, RECIPIENT *,
			               const char *, DSN *);

#endif

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
