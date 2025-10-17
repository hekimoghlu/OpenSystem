/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 29, 2022.
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

#ifndef _INET_ADDR_LIST_H_INCLUDED_
#define _INET_ADDR_LIST_H_INCLUDED_

/*++
/* NAME
/*	inet_addr_list 3h
/* SUMMARY
/*	internet address list manager
/* SYNOPSIS
/*	#include <inet_addr_list.h>
/* DESCRIPTION
/* .nf

 /*
  * Utility library.
  */
#include <myaddrinfo.h>			/* generic name/addr API */

 /*
  * External interface.
  */
typedef struct INET_ADDR_LIST {
    int     used;			/* nr of elements in use */
    int     size;			/* actual list size */
    struct sockaddr_storage *addrs;	/* payload */
} INET_ADDR_LIST;

extern void inet_addr_list_init(INET_ADDR_LIST *);
extern void inet_addr_list_free(INET_ADDR_LIST *);
extern void inet_addr_list_uniq(INET_ADDR_LIST *);
extern void inet_addr_list_append(INET_ADDR_LIST *, struct sockaddr *);

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
