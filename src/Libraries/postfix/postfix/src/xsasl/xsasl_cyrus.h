/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 30, 2023.
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

#ifndef _XSASL_CYRUS_H_INCLUDED_
#define _XSASL_CYRUS_H_INCLUDED_

/*++
/* NAME
/*	xsasl_cyrus 3h
/* SUMMARY
/*	Cyrus SASL plug-in
/* SYNOPSIS
/*	#include <xsasl_cyrus.h>
/* DESCRIPTION
/* .nf

 /*
  * XSASL library.
  */
#include <xsasl.h>

#if defined(USE_SASL_AUTH) && defined(USE_CYRUS_SASL)

 /*
  * SASL protocol interface
  */
#define XSASL_TYPE_CYRUS "cyrus"

extern XSASL_SERVER_IMPL *xsasl_cyrus_server_init(const char *, const char *);
extern XSASL_CLIENT_IMPL *xsasl_cyrus_client_init(const char *, const char *);

 /*
  * Internal definitions for client and server module.
  */
typedef int (*XSASL_CYRUS_CB) (void);

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
