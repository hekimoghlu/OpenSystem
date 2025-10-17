/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 28, 2024.
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
#define SMTPD_NAME_CLIENT	"Client host"
#define SMTPD_NAME_REV_CLIENT	"Unverified Client host"
#define SMTPD_NAME_CCERT	"Client certificate"
#define SMTPD_NAME_SASL_USER	"SASL login name"
#define SMTPD_NAME_HELO		"Helo command"
#define SMTPD_NAME_SENDER	"Sender address"
#define SMTPD_NAME_RECIPIENT	"Recipient address"
#define SMTPD_NAME_ETRN		"Etrn command"
#define SMTPD_NAME_DATA		"Data command"
#define SMTPD_NAME_EOD		"End-of-data"

 /*
  * Workaround for absence of "bad sender address" status code: use "bad
  * sender address syntax" instead. If we were to use "4.1.0" then we would
  * lose the critical distinction between sender and recipient problems.
  */
#define SND_DSN			"4.1.7"

extern const char *smtpd_dsn_fix(const char *, const char *);

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
