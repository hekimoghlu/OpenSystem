/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 20, 2024.
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

#ifndef _VALID_HOSTNAME_H_INCLUDED_
#define _VALID_HOSTNAME_H_INCLUDED_

/*++
/* NAME
/*	valid_hostname 3h
/* SUMMARY
/*	validate hostname
/* SYNOPSIS
/*	#include <valid_hostname.h>
/* DESCRIPTION
/* .nf

 /* External interface */

#define VALID_HOSTNAME_LEN	255	/* RFC 1035 */
#define VALID_LABEL_LEN		63	/* RFC 1035 */

#define DONT_GRIPE		0
#define DO_GRIPE		1

extern int valid_hostname(const char *, int);
extern int valid_hostaddr(const char *, int);
extern int valid_ipv4_hostaddr(const char *, int);
extern int valid_ipv6_hostaddr(const char *, int);
extern int valid_hostport(const char *, int);

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
