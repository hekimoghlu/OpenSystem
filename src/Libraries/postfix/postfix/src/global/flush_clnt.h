/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 24, 2024.
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

#ifndef _FLUSH_CLNT_H_INCLUDED_
#define _FLUSH_CLNT_H_INCLUDED_

/*++
/* NAME
/*	flush_clnt 3h
/* SUMMARY
/*	flush backed up mail
/* SYNOPSIS
/*	#include <flush_clnt.h>
/* DESCRIPTION
/* .nf

 /*
  * External interface.
  */
extern void flush_init(void);
extern int flush_add(const char *, const char *);
extern int flush_send_site(const char *);
extern int flush_send_file(const char *);
extern int flush_refresh(void);
extern int flush_purge(void);

 /*
  * Mail flush server requests.
  */
#define FLUSH_REQ_ADD		"add"	/* append queue ID to site log */
#define FLUSH_REQ_SEND_SITE	"send_site"	/* flush mail for site */
#define FLUSH_REQ_SEND_FILE	"send_file"	/* flush one queue file */
#define FLUSH_REQ_REFRESH	"rfrsh"	/* refresh old logfiles */
#define FLUSH_REQ_PURGE		"purge"	/* refresh all logfiles */

 /*
  * Mail flush server status codes.
  */
#define FLUSH_STAT_FAIL		-1	/* request failed */
#define FLUSH_STAT_OK		0	/* request executed */
#define FLUSH_STAT_BAD		3	/* invalid parameter */
#define FLUSH_STAT_DENY		4	/* request denied */


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
