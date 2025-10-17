/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 22, 2024.
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

#ifndef _QUOTE_822_H_INCLUDED_
#define _QUOTE_822_H_INCLUDED_

/*++
/* NAME
/*	quote_822_local 3h
/* SUMMARY
/*	quote local part of mailbox
/* SYNOPSIS
/*	#include "quote_822_local.h"
/* DESCRIPTION
/* .nf

 /*
  * Utility library.
  */
#include <vstring.h>

 /*
  * Global library.
  */
#include <quote_flags.h>

 /*
  * External interface.
  */
extern VSTRING *quote_822_local_flags(VSTRING *, const char *, int);
extern VSTRING *unquote_822_local(VSTRING *, const char *);
#define quote_822_local(dst, src) \
	quote_822_local_flags((dst), (src), QUOTE_FLAG_DEFAULT)

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
