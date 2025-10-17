/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 14, 2024.
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

#ifndef _DOMAIN_LIST_H_INCLUDED_
#define _DOMAIN_LIST_H_INCLUDED_

/*++
/* NAME
/*	domain_list 3h
/* SUMMARY
/*	match a host or domain name against a pattern list
/* SYNOPSIS
/*	#include <domain_list.h>
/* DESCRIPTION
/* .nf

 /*
  * Utility library.
  */
#include <match_list.h>

 /*
  * External interface.
  */
#define DOMAIN_LIST	MATCH_LIST

#define domain_list_init(o, f, p)\
			match_list_init((o), (f), (p), 1, match_hostname)
#define domain_list_match	match_list_match
#define domain_list_free	match_list_free

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
