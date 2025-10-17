/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 16, 2023.
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

#ifndef _TLS_PROXY_H_INCLUDED_
#define _TLS_PROXY_H_INCLUDED_

/*++
/* NAME
/*	tls_proxy_clnt 3h
/* SUMMARY
/*	postscreen TLS proxy support
/* SYNOPSIS
/*	#include <tls_proxy_clnt.h>
/* DESCRIPTION
/* .nf

 /*
  * Utility library.
  */
#include <vstream.h>
#include <attr.h>

 /*
  * TLS library.
  */
#include <tls.h>

 /*
  * External interface.
  */
#define TLS_PROXY_FLAG_ROLE_SERVER	(1<<0)	/* request server role */
#define TLS_PROXY_FLAG_ROLE_CLIENT	(1<<1)	/* request client role */
#define TLS_PROXY_FLAG_SEND_CONTEXT	(1<<2)	/* send TLS context */

#ifdef USE_TLS

extern VSTREAM *tls_proxy_open(const char *, int, VSTREAM *, const char *,
			               const char *, int);
extern TLS_SESS_STATE *tls_proxy_context_receive(VSTREAM *);
extern void tls_proxy_context_free(TLS_SESS_STATE *);
extern int tls_proxy_context_print(ATTR_PRINT_MASTER_FN, VSTREAM *, int, void *);
extern int tls_proxy_context_scan(ATTR_SCAN_MASTER_FN, VSTREAM *, int, void *);

#endif

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
