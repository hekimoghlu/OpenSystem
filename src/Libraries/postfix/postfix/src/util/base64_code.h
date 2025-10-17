/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 23, 2024.
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

#ifndef _BASE64_CODE_H_INCLUDED_
#define _BASE64_CODE_H_INCLUDED_

/*++
/* NAME
/*	base64_code 3h
/* SUMMARY
/*	encode/decode data, base 64 style
/* SYNOPSIS
/*	#include <base64_code.h>
/* DESCRIPTION
/* .nf

 /*
  * Utility library.
 */
#include <vstring.h>

 /*
  * External interface.
  */
extern VSTRING *base64_encode_opt(VSTRING *, const char *, ssize_t, int);
extern VSTRING *WARN_UNUSED_RESULT base64_decode_opt(VSTRING *, const char *, ssize_t, int);

#define BASE64_FLAG_NONE	0
#define BASE64_FLAG_APPEND	(1<<0)

#define base64_encode(bp, cp, ln) \
	base64_encode_opt((bp), (cp), (ln), BASE64_FLAG_NONE)
#define base64_decode(bp, cp, ln) \
	base64_decode_opt((bp), (cp), (ln), BASE64_FLAG_NONE)

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
