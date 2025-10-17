/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 12, 2022.
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

#ifndef _DICT_PROXY_H_INCLUDED_
#define _DICT_PROXY_H_INCLUDED_

/*++
/* NAME
/*	dict_proxy 3h
/* SUMMARY
/*	dictionary manager interface to PROXY maps
/* SYNOPSIS
/*	#include <dict_proxy.h>
/* DESCRIPTION
/* .nf

 /*
  * Utility library.
  */
#include <dict.h>

 /*
  * External interface.
  */
#define DICT_TYPE_PROXY	"proxy"

extern DICT *dict_proxy_open(const char *, int, int);

 /*
  * Protocol interface.
  */
#define PROXY_REQ_OPEN		"open"
#define PROXY_REQ_LOOKUP	"lookup"
#define PROXY_REQ_UPDATE	"update"
#define PROXY_REQ_DELETE	"delete"
#define PROXY_REQ_SEQUENCE	"sequence"

#define PROXY_STAT_OK		0	/* operation succeeded */
#define PROXY_STAT_NOKEY	1	/* requested key not found */
#define PROXY_STAT_RETRY	2	/* try lookup again later */
#define PROXY_STAT_BAD		3	/* invalid request parameter */
#define PROXY_STAT_DENY		4	/* table not approved for proxying */
#define PROXY_STAT_CONFIG	5	/* DICT_ERR_CONFIG error */

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
