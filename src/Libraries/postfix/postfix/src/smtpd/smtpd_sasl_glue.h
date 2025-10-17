/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 22, 2022.
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
extern void smtpd_sasl_state_init(SMTPD_STATE *);
extern void smtpd_sasl_initialize(void);
extern void smtpd_sasl_activate(SMTPD_STATE *, const char *, const char *);
extern void smtpd_sasl_deactivate(SMTPD_STATE *);
extern int smtpd_sasl_authenticate(SMTPD_STATE *, const char *, const char *);
extern void smtpd_sasl_login(SMTPD_STATE *, const char *, const char *);
extern void smtpd_sasl_logout(SMTPD_STATE *);
extern int permit_sasl_auth(SMTPD_STATE *, int, int);

#define smtpd_sasl_is_active(s)		((s)->sasl_server != 0)
#define smtpd_sasl_set_inactive(s)	((void) ((s)->sasl_server = 0))

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
