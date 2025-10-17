/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 1, 2024.
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
#include <sys/time.h>

 /*
  * Utility library.
  */
#include <vstream.h>
#include <vstring.h>

 /*
  * Global library.
  */
#include <mail_stream.h>

 /*
  * Per-session state.
  */
typedef struct {
    int     err;			/* error flags */
    VSTREAM *client;			/* client connection */
    VSTRING *message;			/* message buffer */
    VSTRING *buf;			/* line buffer */
    struct timeval arrival_time;	/* start of session */
    char   *name;			/* client name */
    char   *addr;			/* client IP address */
    char   *port;			/* client TCP port */
    char   *namaddr;			/* name[addr]:port */
    char   *rfc_addr;			/* RFC 2821 client IP address */
    int     addr_family;		/* address family */
    char   *queue_id;			/* queue file ID */
    VSTREAM *cleanup;			/* cleanup server */
    MAIL_STREAM *dest;			/* cleanup server */
    int     rcpt_count;			/* recipient count */
    char   *reason;			/* exception name */
    char   *sender;			/* sender address */
    char   *recipient;			/* recipient address */
    char   *protocol;			/* protocol name */
    char   *where;			/* protocol state */
    VSTRING *why_rejected;		/* REJECT reason */
} QMQPD_STATE;

 /*
  * Representation of unknown upstream client or message information within
  * qmqpd processes. This is not the representation that Postfix uses in
  * queue files, in queue manager delivery requests, or in XCLIENT/XFORWARD
  * commands!
  */
#define CLIENT_ATTR_UNKNOWN	"unknown"

#define CLIENT_NAME_UNKNOWN	CLIENT_ATTR_UNKNOWN
#define CLIENT_ADDR_UNKNOWN	CLIENT_ATTR_UNKNOWN
#define CLIENT_PORT_UNKNOWN	CLIENT_ATTR_UNKNOWN
#define CLIENT_NAMADDR_UNKNOWN	CLIENT_ATTR_UNKNOWN

 /*
  * QMQP protocol status codes.
  */
#define QMQPD_STAT_OK		'K'
#define QMQPD_STAT_RETRY	'Z'
#define QMQPD_STAT_HARD		'D'

 /*
  * qmqpd_state.c
  */
QMQPD_STATE *qmqpd_state_alloc(VSTREAM *);
void    qmqpd_state_free(QMQPD_STATE *);

 /*
  * qmqpd_peer.c
  */
void    qmqpd_peer_init(QMQPD_STATE *);
void    qmqpd_peer_reset(QMQPD_STATE *);

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
