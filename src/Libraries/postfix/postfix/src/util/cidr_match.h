/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 25, 2025.
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

#ifndef _CIDR_MATCH_H_INCLUDED_
#define _CIDR_MATCH_H_INCLUDED_

/*++
/* NAME
/*	dict_cidr 3h
/* SUMMARY
/*	CIDR-style pattern matching
/* SYNOPSIS
/*	#include <cidr_match.h>
/* DESCRIPTION
/* .nf

 /*
  * System library.
  */
#include <limits.h>			/* CHAR_BIT */

 /*
  * Utility library.
  */
#include <myaddrinfo.h>			/* MAI_V6ADDR_BYTES etc. */
#include <vstring.h>

 /*
  * External interface.
  * 
  * Address length is protocol dependent. Find out how large our address byte
  * strings should be.
  */
#ifdef HAS_IPV6
# define CIDR_MATCH_ABYTES	MAI_V6ADDR_BYTES
#else
# define CIDR_MATCH_ABYTES	MAI_V4ADDR_BYTES
#endif

 /*
  * Each parsed CIDR pattern can be member of a linked list.
  */
typedef struct CIDR_MATCH {
    int     op;				/* operation, match or control flow */
    int     match;			/* positive or negative match */
    unsigned char net_bytes[CIDR_MATCH_ABYTES];	/* network portion */
    unsigned char mask_bytes[CIDR_MATCH_ABYTES];	/* network mask */
    unsigned char addr_family;		/* AF_XXX */
    unsigned char addr_byte_count;	/* typically, 4 or 16 */
    unsigned char addr_bit_count;	/* optimization */
    unsigned char mask_shift;		/* optimization */
    struct CIDR_MATCH *next;		/* next entry */
    struct CIDR_MATCH *block_end;	/* block terminator */
} CIDR_MATCH;

#define CIDR_MATCH_OP_MATCH	1	/* Match this pattern */
#define CIDR_MATCH_OP_IF	2	/* Increase if/endif nesting on match */
#define CIDR_MATCH_OP_ENDIF	3	/* Decrease if/endif nesting on match */

#define CIDR_MATCH_TRUE		1	/* Request positive match */
#define CIDR_MATCH_FALSE	0	/* Request negative match */

extern VSTRING *cidr_match_parse(CIDR_MATCH *, char *, int, VSTRING *);
extern VSTRING *cidr_match_parse_if(CIDR_MATCH *, char *, int, VSTRING *);
extern void cidr_match_endif(CIDR_MATCH *);

extern CIDR_MATCH *cidr_match_execute(CIDR_MATCH *, const char *);

/* LICENSE
/* .ad
/* .fi
/*	The Secure Mailer license must be distributed with this software.
/* AUTHOR(S)
/*	Wietse Venema
/*	IBM T.J. Watson Research
/*	P.O. Box 704
/*	Yorktown Heights, NY 10598, USA
/*
/*	Wietse Venema
/*	Google, Inc.
/*	111 8th Avenue
/*	New York, NY 10011, USA
/*--*/

#endif
