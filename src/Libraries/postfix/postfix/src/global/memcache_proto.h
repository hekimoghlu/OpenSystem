/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 21, 2022.
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

#ifndef _MEMCACHE_PROTO_H_INCLUDED_
#define _MEMCACHE_PROTO_H_INCLUDED_

/*++
/* NAME
/*	memcache_proto 3h
/* SUMMARY
/*	memcache low-level protocol
/* SYNOPSIS
/*	#include <memcache_proto.h>
/* DESCRIPTION
/* .nf

 /*
  * External interface.
  */
extern int memcache_get(VSTREAM *, VSTRING *, ssize_t);
extern int memcache_vprintf(VSTREAM *, const char *, va_list);
extern int PRINTFLIKE(2, 3) memcache_printf(VSTREAM *, const char *fmt,...);
extern int memcache_fread(VSTREAM *, VSTRING *, ssize_t);
extern int memcache_fwrite(VSTREAM *, const char *, ssize_t);

/* LICENSE
/* .ad
/* .fi
/*	The Secure Mailer license must be distributed with this software.
/*	AUTHOR(S)
/*	Wietse Venema
/*	IBM T.J. Watson Research
/*	P.O. Box 704
/*	Yorktown Heights, NY 10598, USA
/*--*/

#endif
