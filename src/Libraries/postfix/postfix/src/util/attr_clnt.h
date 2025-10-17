/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 12, 2025.
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

#ifndef _ATTR_CLNT_H_INCLUDED_
#define _ATTR_CLNT_H_INCLUDED_

/*++
/* NAME
/*	attr_clnt 3h
/* SUMMARY
/*	attribute query-reply client
/* SYNOPSIS
/*	#include <attr_clnt.h>
/* DESCRIPTION
/* .nf

 /*
  * System library.
  */
#include <stdarg.h>

 /*
  * Utility library.
  */
#include <attr.h>

 /*
  * External interface.
  */
typedef struct ATTR_CLNT ATTR_CLNT;
typedef int (*ATTR_CLNT_PRINT_FN) (VSTREAM *, int, va_list);
typedef int (*ATTR_CLNT_SCAN_FN) (VSTREAM *, int, va_list);

extern ATTR_CLNT *attr_clnt_create(const char *, int, int, int);
extern int attr_clnt_request(ATTR_CLNT *, int,...);
extern void attr_clnt_free(ATTR_CLNT *);
extern void attr_clnt_control(ATTR_CLNT *, int,...);

#define ATTR_CLNT_CTL_END	0
#define ATTR_CLNT_CTL_PROTO	1	/* print/scan functions */
#define ATTR_CLNT_CTL_REQ_LIMIT	2	/* requests per connection */
#define ATTR_CLNT_CTL_TRY_LIMIT	3	/* attempts per request */
#define ATTR_CLNT_CTL_TRY_DELAY	4	/* pause between requests */

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
