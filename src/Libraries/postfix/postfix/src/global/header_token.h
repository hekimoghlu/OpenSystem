/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 9, 2023.
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

#ifndef _HEADER_TOKEN_H_INCLUDED_
#define _HEADER_TOKEN_H_INCLUDED_

/*++
/* NAME
/*	header_token 3h
/* SUMMARY
/*	mail header parser
/* SYNOPSIS
/*	#include "header_token.h"
 DESCRIPTION
 .nf

 /*
  * Utility library.
  */
#include <vstring.h>

 /*
  * HEADER header parser tokens. Specials and controls are represented by
  * themselves. Character pointers point to substrings in a token buffer.
  */
typedef struct HEADER_TOKEN {
    int     type;			/* see below */
    union {
	const char *value;		/* just a pointer, not a copy */
	ssize_t offset;			/* index into token buffer */
    }       u;				/* indent beats any alternative */
} HEADER_TOKEN;

#define HEADER_TOK_TOKEN	256
#define HEADER_TOK_QSTRING	257

extern ssize_t header_token(HEADER_TOKEN *, ssize_t, VSTRING *, const char **, const char *, int);

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
