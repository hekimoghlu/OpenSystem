/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 15, 2023.
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

#ifndef _MAC_EXPAND_H_INCLUDED_
#define _MAC_EXPAND_H_INCLUDED_

/*++
/* NAME
/*	mac_expand 3h
/* SUMMARY
/*	expand macro references in string
/* SYNOPSIS
/*	#include <mac_expand.h>
/* DESCRIPTION
/* .nf

 /*
  * Utility library.
  */
#include <vstring.h>
#include <mac_parse.h>

 /*
  * Features.
  */
#define MAC_EXP_FLAG_NONE	(0)
#define MAC_EXP_FLAG_RECURSE	(1<<0)
#define MAC_EXP_FLAG_APPEND	(1<<1)
#define MAC_EXP_FLAG_SCAN	(1<<2)
#define MAC_EXP_FLAG_PRINTABLE  (1<<3)

 /*
  * Real lookup or just a test?
  */
#define MAC_EXP_MODE_TEST	(0)
#define MAC_EXP_MODE_USE	(1)

typedef const char *(*MAC_EXP_LOOKUP_FN) (const char *, int, void *);

extern int mac_expand(VSTRING *, const char *, int, const char *, MAC_EXP_LOOKUP_FN, void *);

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
