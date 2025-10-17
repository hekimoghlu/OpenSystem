/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 19, 2021.
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

#ifndef _RESOLVE_CLNT_H_INCLUDED_
#define _RESOLVE_CLNT_H_INCLUDED_

/*++
/* NAME
/*	resolve_clnt 3h
/* SUMMARY
/*	address resolver client
/* SYNOPSIS
/*	#include <resolve_clnt.h>
/* DESCRIPTION
/* .nf

 /*
  * Utility library.
  */
#include <vstring.h>

 /*
  * External interface.
  */
#define RESOLVE_REGULAR	"resolve"
#define RESOLVE_VERIFY	"verify"

#define RESOLVE_FLAG_FINAL	(1<<0)	/* final delivery */
#define RESOLVE_FLAG_ROUTED	(1<<1)	/* routed destination */
#define RESOLVE_FLAG_ERROR	(1<<2)	/* bad destination syntax */
#define RESOLVE_FLAG_FAIL	(1<<3)	/* request failed */

#define RESOLVE_CLASS_LOCAL	(1<<8)	/* mydestination/inet_interfaces */
#define RESOLVE_CLASS_ALIAS	(1<<9)	/* virtual_alias_domains */
#define RESOLVE_CLASS_VIRTUAL	(1<<10)	/* virtual_mailbox_domains */
#define RESOLVE_CLASS_RELAY	(1<<11)	/* relay_domains */
#define RESOLVE_CLASS_DEFAULT	(1<<12)	/* raise reject_unauth_destination */

#define RESOLVE_CLASS_FINAL \
	(RESOLVE_CLASS_LOCAL | RESOLVE_CLASS_ALIAS | RESOLVE_CLASS_VIRTUAL)

#define RESOLVE_CLASS_MASK \
	(RESOLVE_CLASS_LOCAL | RESOLVE_CLASS_ALIAS | RESOLVE_CLASS_VIRTUAL \
	| RESOLVE_CLASS_RELAY | RESOLVE_CLASS_DEFAULT)

typedef struct RESOLVE_REPLY {
    VSTRING *transport;
    VSTRING *nexthop;
    VSTRING *recipient;
    int     flags;
} RESOLVE_REPLY;

extern void resolve_clnt_init(RESOLVE_REPLY *);
extern void resolve_clnt(const char *, const char *, const char *, RESOLVE_REPLY *);
extern void resolve_clnt_free(RESOLVE_REPLY *);

#define RESOLVE_NULL_FROM	""

#define resolve_clnt_query_from(f, a, r) \
	resolve_clnt(RESOLVE_REGULAR, (f), (a), (r))
#define resolve_clnt_verify_from(f, a, r) \
	resolve_clnt(RESOLVE_VERIFY, (f), (a), (r))

#define RESOLVE_CLNT_ASSIGN(reply, transport, nexthop, recipient) { \
	(reply).transport = (transport); \
	(reply).nexthop = (nexthop); \
	(reply).recipient = (recipient); \
    }

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
