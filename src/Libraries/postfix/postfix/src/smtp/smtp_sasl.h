/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 18, 2022.
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
extern void smtp_sasl_initialize(void);
extern void smtp_sasl_connect(SMTP_SESSION *);
extern int smtp_sasl_passwd_lookup(SMTP_SESSION *);
extern void smtp_sasl_start(SMTP_SESSION *, const char *, const char *);
extern int smtp_sasl_authenticate(SMTP_SESSION *, DSN_BUF *);
extern void smtp_sasl_cleanup(SMTP_SESSION *);

extern void smtp_sasl_helo_auth(SMTP_SESSION *, const char *);
extern int smtp_sasl_helo_login(SMTP_STATE *);

extern void smtp_sasl_passivate(SMTP_SESSION *, VSTRING *);
extern int smtp_sasl_activate(SMTP_SESSION *, char *);
extern STRING_LIST *smtp_sasl_mechs;

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
