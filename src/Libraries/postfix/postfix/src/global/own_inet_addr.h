/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 29, 2025.
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

#ifndef _OWN_INET_ADDR_H_INCLUDED_
#define _OWN_INET_ADDR_H_INCLUDED_

/*++
/* NAME
/*	own_inet_addr 3h
/* SUMMARY
/*	determine if IP address belongs to this mail system instance
/* SYNOPSIS
/*	#include <own_inet_addr.h>
/* DESCRIPTION
/* .nf

 /*
  * Utility library.
  */
#include <inet_addr_list.h>

 /*
  * External interface.
  */
extern int own_inet_addr(struct sockaddr *);
extern struct INET_ADDR_LIST *own_inet_addr_list(void);
extern struct INET_ADDR_LIST *own_inet_mask_list(void);
extern int proxy_inet_addr(struct sockaddr *);
extern struct INET_ADDR_LIST *proxy_inet_addr_list(void);

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
