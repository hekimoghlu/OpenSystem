/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 3, 2025.
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

#ifndef _INET_PROTO_INFO_H_INCLUDED_
#define _INET_PROTO_INFO_H_INCLUDED_

/*++
/* NAME
/*	inet_proto_info 3h
/* SUMMARY
/*	convert protocol names to assorted constants
/* SYNOPSIS
/*	#include <inet_proto_info.h>
 DESCRIPTION
 .nf

 /*
  * External interface.
  */
typedef struct {
    unsigned int ai_family;		/* PF_UNSPEC, PF_INET, or PF_INET6 */
    unsigned int *ai_family_list;	/* PF_INET and/or PF_INET6 */
    unsigned int *dns_atype_list;	/* TAAAA and/or TA */
    unsigned char *sa_family_list;	/* AF_INET6 and/or AF_INET */
} INET_PROTO_INFO;

 /*
  * Some compilers won't link initialized data unless we call a function in
  * the same source file. Therefore, inet_proto_info() is a function instead
  * of a global variable.
  */
#define inet_proto_info() \
    (inet_proto_table ? inet_proto_table : \
	inet_proto_init("default protocol setting", DEF_INET_PROTOCOLS))

extern INET_PROTO_INFO *inet_proto_init(const char *, const char *);
extern INET_PROTO_INFO *inet_proto_table;

#define INET_PROTO_NAME_IPV6	"ipv6"
#define INET_PROTO_NAME_IPV4	"ipv4"
#define INET_PROTO_NAME_ALL	"all"

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
