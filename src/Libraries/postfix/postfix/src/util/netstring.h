/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 25, 2024.
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

#ifndef _NETSTRING_H_INCLUDED_
#define _NETSTRING_H_INCLUDED_

/*++
/* NAME
/*	netstring 3h
/* SUMMARY
/*	netstring stream I/O support
/* SYNOPSIS
/*	#include <netstring.h>
/* DESCRIPTION
/* .nf

 /*
  * Utility library.
  */
#include <vstring.h>
#include <vstream.h>

 /*
  * External interface.
  */
#define NETSTRING_ERR_EOF	1	/* unexpected disconnect */
#define NETSTRING_ERR_TIME	2	/* time out */
#define NETSTRING_ERR_FORMAT	3	/* format error */
#define NETSTRING_ERR_SIZE	4	/* netstring too large */

extern void netstring_except(VSTREAM *, int);
extern void netstring_setup(VSTREAM *, int);
extern ssize_t netstring_get_length(VSTREAM *);
extern VSTRING *netstring_get_data(VSTREAM *, VSTRING *, ssize_t);
extern void netstring_get_terminator(VSTREAM *);
extern VSTRING *netstring_get(VSTREAM *, VSTRING *, ssize_t);
extern void netstring_put(VSTREAM *, const char *, ssize_t);
extern void netstring_put_multi(VSTREAM *,...);
extern void netstring_fflush(VSTREAM *);
extern VSTRING *netstring_memcpy(VSTRING *, const char *, ssize_t);
extern VSTRING *netstring_memcat(VSTRING *, const char *, ssize_t);
extern const char *netstring_strerror(int);

#define NETSTRING_PUT_BUF(str, buf) \
	netstring_put((str), vstring_str(buf), VSTRING_LEN(buf))

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
