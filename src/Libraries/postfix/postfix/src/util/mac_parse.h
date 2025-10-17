/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 31, 2024.
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

#ifndef _MAC_PARSE_H_INCLUDED_
#define _MAC_PARSE_H_INCLUDED_

/*++
/* NAME
/*	mac_parse 3h
/* SUMMARY
/*	locate macro references in string
/* SYNOPSIS
/*	#include <mac_parse.h>
/* DESCRIPTION
/* .nf

 /*
  * Utility library.
  */
#include <vstring.h>

 /*
  * External interface.
  */
#define MAC_PARSE_LITERAL	1
#define MAC_PARSE_EXPR		2
#define MAC_PARSE_VARNAME	MAC_PARSE_EXPR	/* 2.1 compatibility */

#define MAC_PARSE_OK		0
#define MAC_PARSE_ERROR		(1<<0)
#define MAC_PARSE_UNDEF		(1<<1)
#define MAC_PARSE_USER		2	/* start user definitions */

typedef int (*MAC_PARSE_FN) (int, VSTRING *, void *);

extern int WARN_UNUSED_RESULT mac_parse(const char *, MAC_PARSE_FN, void *);

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
