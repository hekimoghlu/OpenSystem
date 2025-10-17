/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 1, 2023.
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

#ifndef _TLS_SCACHE_H_INCLUDED_
#define _TLS_SCACHE_H_INCLUDED_

/*++
/* NAME
/*	tls_scache 3h
/* SUMMARY
/*	TLS session cache manager
/* SYNOPSIS
/*	#include <tls_scache.h>
/* DESCRIPTION
/* .nf

 /*
  * Utility library.
  */
#include <dict.h>
#include <vstring.h>

 /*
  * External interface.
  */
typedef struct {
    int     flags;			/* see below */
    DICT   *db;				/* database handle */
    char   *cache_label;		/* "smtpd", "smtp" or "lmtp" */
    int     verbose;			/* enable verbose logging */
    int     timeout;			/* smtp(d)_tls_session_cache_timeout */
    char   *saved_cursor;		/* cursor cache ID */
} TLS_SCACHE;

#define TLS_TICKET_NAMELEN	16	/* RFC 5077 ticket key name length */
#define TLS_TICKET_IVLEN	16	/* RFC 5077 ticket IV length */
#define TLS_TICKET_KEYLEN	32	/* AES-256-CBC key size */
#define TLS_TICKET_MACLEN	32	/* RFC 5077 HMAC key size */
#define TLS_SESSION_LIFEMIN	120	/* May you live to 120! */

typedef struct TLS_TICKET_KEY {
    unsigned char name[TLS_TICKET_NAMELEN];
    unsigned char bits[TLS_TICKET_KEYLEN];
    unsigned char hmac[TLS_TICKET_MACLEN];
    time_t  tout;
} TLS_TICKET_KEY;

#define TLS_SCACHE_FLAG_DEL_SAVED_CURSOR	(1<<0)

extern TLS_SCACHE *tls_scache_open(const char *, const char *, int, int);
extern void tls_scache_close(TLS_SCACHE *);
extern int tls_scache_lookup(TLS_SCACHE *, const char *, VSTRING *);
extern int tls_scache_update(TLS_SCACHE *, const char *, const char *, ssize_t);
extern int tls_scache_delete(TLS_SCACHE *, const char *);
extern int tls_scache_sequence(TLS_SCACHE *, int, char **, VSTRING *);
extern TLS_TICKET_KEY *tls_scache_key(unsigned char *, time_t, int);
extern TLS_TICKET_KEY *tls_scache_key_rotate(TLS_TICKET_KEY *);

#define TLS_SCACHE_DONT_NEED_CACHE_ID		((char **) 0)
#define TLS_SCACHE_DONT_NEED_SESSION		((VSTRING *) 0)

#define TLS_SCACHE_SEQUENCE_NOTHING \
	TLS_SCACHE_DONT_NEED_CACHE_ID, TLS_SCACHE_DONT_NEED_SESSION

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
