/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 16, 2023.
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
#include <vstream.h>
#include <nbbio.h>

 /*
  * TLS library.
  */
#include <tls.h>

 /*
  * Internal interface.
  */
typedef struct {
    int     flags;			/* see below */
    int     req_flags;			/* request flags, see tls_proxy.h */
    char   *service;			/* argv[0] */
    VSTREAM *plaintext_stream;		/* local peer: postscreen(8), etc. */
    NBBIO  *plaintext_buf;		/* plaintext buffer */
    int     ciphertext_fd;		/* remote peer */
    EVENT_NOTIFY_FN ciphertext_timer;	/* kludge */
    int     timeout;			/* read/write time limit */
    char   *remote_endpt;		/* printable remote endpoint */
    char   *server_id;			/* cache management */
    TLS_SESS_STATE *tls_context;	/* llibtls state */
    int     ssl_last_err;		/* TLS I/O state */
} TLSP_STATE;

#define TLSP_FLAG_DO_HANDSHAKE	(1<<0)

extern TLSP_STATE *tlsp_state_create(const char *, VSTREAM *);
extern void tlsp_state_free(TLSP_STATE *);

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
