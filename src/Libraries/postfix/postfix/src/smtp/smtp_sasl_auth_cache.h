/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 13, 2025.
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

#ifndef _SMTP_SASL_AUTH_CACHE_H_INCLUDED_
#define _SMTP_SASL_AUTH_CACHE_H_INCLUDED_

/*++
/* NAME
/*	smtp_sasl_auth_cache 3h
/* SUMMARY
/*	Postfix SASL authentication failure cache
/* SYNOPSIS
/*	#include "smtp.h"
/*	#include "smtp_sasl_auth_cache.h"
/* DESCRIPTION
/* .nf

 /*
  * Utility library.
  */
#include <dict.h>

 /*
  * This code stores hashed passwords which requires OpenSSL.
  */
#if defined(USE_TLS) && defined(USE_SASL_AUTH)
#define HAVE_SASL_AUTH_CACHE

 /*
  * External interface.
  */
typedef struct {
    DICT   *dict;
    int     ttl;
    char   *dsn;
    char   *text;
} SMTP_SASL_AUTH_CACHE;

extern SMTP_SASL_AUTH_CACHE *smtp_sasl_auth_cache_init(const char *, int);
extern void smtp_sasl_auth_cache_store(SMTP_SASL_AUTH_CACHE *, const SMTP_SESSION *, const SMTP_RESP *);
extern int smtp_sasl_auth_cache_find(SMTP_SASL_AUTH_CACHE *, const SMTP_SESSION *);

#define smtp_sasl_auth_cache_dsn(cp)	((cp)->dsn)
#define smtp_sasl_auth_cache_text(cp)	((cp)->text)

#endif

/* LICENSE
/* .ad
/* .fi
/*	The Secure Mailer license must be distributed with this software.
/* AUTHOR(S)
/*	Initial implementation by:
/*	Till Franke
/*	SuSE Rhein/Main AG
/*	65760 Eschborn, Germany
/*
/*	Adopted by:
/*	Wietse Venema
/*	IBM T.J. Watson Research
/*	P.O. Box 704
/*	Yorktown Heights, NY 10598, USA
/*--*/

#endif
