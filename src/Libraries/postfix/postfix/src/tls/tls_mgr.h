/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 28, 2024.
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

#ifndef _TLS_MGR_CLNT_H_INCLUDED_
#define _TLS_MGR_CLNT_H_INCLUDED_

/*++
/* NAME
/*	tls_mgr 3h
/* SUMMARY
/*	tlsmgr client interface
/* SYNOPSIS
/*	#include <tls_mgr.h>
/* DESCRIPTION
/* .nf

 /*
  * TLS library
  */
#include <tls_scache.h>			/* Session ticket keys */

 /*
  * TLS manager protocol.
  */
#define TLS_MGR_SERVICE		"tlsmgr"
#define TLS_MGR_CLASS		"private"

#define TLS_MGR_ATTR_REQ	"request"
#define TLS_MGR_REQ_SEED	"seed"
#define TLS_MGR_REQ_POLICY	"policy"
#define TLS_MGR_REQ_LOOKUP	"lookup"
#define TLS_MGR_REQ_UPDATE	"update"
#define TLS_MGR_REQ_DELETE	"delete"
#define TLS_MGR_REQ_TKTKEY	"tktkey"
#define TLS_MGR_ATTR_CACHABLE	"cachable"
#define TLS_MGR_ATTR_CACHE_TYPE	"cache_type"
#define TLS_MGR_ATTR_SEED	"seed"
#define TLS_MGR_ATTR_CACHE_ID	"cache_id"
#define TLS_MGR_ATTR_SESSION	"session"
#define TLS_MGR_ATTR_SIZE	"size"
#define TLS_MGR_ATTR_STATUS	"status"
#define TLS_MGR_ATTR_KEYNAME	"keyname"
#define TLS_MGR_ATTR_KEYBUF	"keybuf"
#define TLS_MGR_ATTR_SESSTOUT	"timeout"

 /*
  * TLS manager request status codes.
  */
#define TLS_MGR_STAT_OK		0	/* success */
#define TLS_MGR_STAT_ERR	(-1)	/* object not found */
#define TLS_MGR_STAT_FAIL	(-2)	/* protocol error */

 /*
  * Functional interface.
  */
extern int tls_mgr_seed(VSTRING *, int);
extern int tls_mgr_policy(const char *, int *, int *);
extern int tls_mgr_lookup(const char *, const char *, VSTRING *);
extern int tls_mgr_update(const char *, const char *, const char *, ssize_t);
extern int tls_mgr_delete(const char *, const char *);
extern TLS_TICKET_KEY *tls_mgr_key(unsigned char *, int);

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
