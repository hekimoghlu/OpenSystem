/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 30, 2025.
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

#ifndef _ABOUNCE_H_INCLUDED_
#define _ABOUNCE_H_INCLUDED_

/*++
/* NAME
/*	abounce 3h
/* SUMMARY
/*	asynchronous bounce/defer service client
/* SYNOPSIS
/*	#include <abounce.h>
/* DESCRIPTION
/* .nf

 /*
  * Global library.
  */
#include <bounce.h>

 /*
  * Client interface.
  */
typedef void (*ABOUNCE_FN) (int, void *);

extern void abounce_flush(int, const char *, const char *, const char *, int, const char *, const char *, int, ABOUNCE_FN, void *);
extern void adefer_flush(int, const char *, const char *, const char *, int, const char *, const char *, int, ABOUNCE_FN, void *);
extern void adefer_warn(int, const char *, const char *, const char *, int, const char *, const char *, int, ABOUNCE_FN, void *);
extern void atrace_flush(int, const char *, const char *, const char *, int, const char *, const char *, int, ABOUNCE_FN, void *);

extern void abounce_flush_verp(int, const char *, const char *, const char *, int, const char *, const char *, int, const char *, ABOUNCE_FN, void *);
extern void adefer_flush_verp(int, const char *, const char *, const char *, int, const char *, const char *, int, const char *, ABOUNCE_FN, void *);

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
