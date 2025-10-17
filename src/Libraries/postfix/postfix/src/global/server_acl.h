/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 15, 2024.
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

#ifndef _SERVER_ACL_INCLUDED_
#define _SERVER_ACL_INCLUDED_

/*++
/* NAME
/*	dict_memcache 3h
/* SUMMARY
/*	dictionary interface to memcache databases
/* SYNOPSIS
/*	#include <dict_memcache.h>
/* DESCRIPTION
/* .nf

 /*
  * Utility library.
  */
#include <argv.h>

 /*
  * External interface.
  */
typedef ARGV SERVER_ACL;
extern void server_acl_pre_jail_init(const char *, const char *);
extern SERVER_ACL *server_acl_parse(const char *, const char *);
extern int server_acl_eval(const char *, SERVER_ACL *, const char *);

#define SERVER_ACL_NAME_WL_MYNETWORKS "permit_mynetworks"
#define SERVER_ACL_NAME_PERMIT	"permit"
#define SERVER_ACL_NAME_DUNNO	"dunno"
#define SERVER_ACL_NAME_REJECT	"reject"
#define SERVER_ACL_NAME_ERROR	"error"

#define SERVER_ACL_ACT_PERMIT	1
#define SERVER_ACL_ACT_DUNNO	0
#define SERVER_ACL_ACT_REJECT	(-1)
#define SERVER_ACL_ACT_ERROR	(-2)

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
