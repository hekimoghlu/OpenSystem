/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 26, 2024.
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

#ifndef _VRFY_CLNT_H_INCLUDED_
#define _VRFY_CLNT_H_INCLUDED_

/*++
/* NAME
/*	verify_clnt 3h
/* SUMMARY
/*	address verification client interface
/* SYNOPSIS
/*	#include <verify_clnt.h>
/* DESCRIPTION
/* .nf

 /*
  * System library.
  */
#include <stdarg.h>

 /*
  * Global library.
  */
#include <deliver_request.h>

 /*
  * Address verification requests.
  */
#define VRFY_REQ_QUERY		"query"
#define VRFY_REQ_UPDATE		"update"

 /*
  * Request (NOT: address) status codes.
  */
#define VRFY_STAT_OK		0
#define VRFY_STAT_FAIL		(-1)
#define VRFY_STAT_BAD		(-2)

 /*
  * Functional interface.
  */
extern int verify_clnt_query(const char *, int *, VSTRING *);
extern int verify_clnt_update(const char *, int, const char *);

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
