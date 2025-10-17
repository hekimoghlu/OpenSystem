/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 13, 2022.
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
#include <vstring.h>

 /*
  * Application-specific.
  */
typedef int PRINTFPTRLIKE(3, 4) (*SMTPD_PROXY_CMD_FN) (SMTPD_STATE *, int, const char *,...);
typedef int PRINTFPTRLIKE(3, 4) (*SMTPD_PROXY_REC_FPRINTF_FN) (VSTREAM *, int, const char *,...);
typedef int (*SMTPD_PROXY_REC_PUT_FN) (VSTREAM *, int, const char *, ssize_t);

typedef struct SMTPD_PROXY {
    /* Public. */
    VSTREAM *stream;
    VSTRING *request;			/* proxy request buffer */
    VSTRING *reply;			/* proxy reply buffer */
    SMTPD_PROXY_CMD_FN cmd;
    SMTPD_PROXY_REC_FPRINTF_FN rec_fprintf;
    SMTPD_PROXY_REC_PUT_FN rec_put;
    /* Private. */
    int     flags;
    VSTREAM *service_stream;
    const char *service_name;
    int     timeout;
    const char *ehlo_name;
    const char *mail_from;
} SMTPD_PROXY;

#define SMTPD_PROXY_FLAG_SPEED_ADJUST	(1<<0)

#define SMTPD_PROXY_NAME_SPEED_ADJUST	"speed_adjust"

#define SMTPD_PROX_WANT_BAD	0xff	/* Do not use */
#define SMTPD_PROX_WANT_NONE	'\0'	/* Do not receive reply */
#define SMTPD_PROX_WANT_ANY	'0'	/* Expect any reply */
#define SMTPD_PROX_WANT_OK	'2'	/* Expect 2XX reply */
#define SMTPD_PROX_WANT_MORE	'3'	/* Expect 3XX reply */

extern int smtpd_proxy_create(SMTPD_STATE *, int, const char *, int, const char *, const char *);
extern void smtpd_proxy_close(SMTPD_STATE *);
extern void smtpd_proxy_free(SMTPD_STATE *);
extern int smtpd_proxy_parse_opts(const char *, const char *);

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
