/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 19, 2025.
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
extern void smtpd_check_init(void);
extern int smtpd_check_addr(const char *, const char *, int);
extern char *smtpd_check_rewrite(SMTPD_STATE *);
extern char *smtpd_check_client(SMTPD_STATE *);
extern char *smtpd_check_helo(SMTPD_STATE *, char *);
extern char *smtpd_check_mail(SMTPD_STATE *, char *);
extern char *smtpd_check_size(SMTPD_STATE *, off_t);
extern char *smtpd_check_queue(SMTPD_STATE *);
extern char *smtpd_check_rcpt(SMTPD_STATE *, char *);
extern char *smtpd_check_etrn(SMTPD_STATE *, char *);
extern char *smtpd_check_data(SMTPD_STATE *);
extern char *smtpd_check_eod(SMTPD_STATE *);
extern char *smtpd_check_policy(SMTPD_STATE *, char *);

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
