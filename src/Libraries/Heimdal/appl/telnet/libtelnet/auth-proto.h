/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 5, 2022.
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
/*
 * Copyright (C) 1990 by the Massachusetts Institute of Technology
 *
 * Export of this software from the United States of America is assumed
 * to require a specific license from the United States Government.
 * It is the responsibility of any person or organization contemplating
 * export to obtain such a license before exporting.
 *
 * WITHIN THAT CONSTRAINT, permission to use, copy, modify, and
 * distribute this software and its documentation for any purpose and
 * without fee is hereby granted, provided that the above copyright
 * notice appear in all copies and that both that copyright notice and
 * this permission notice appear in supporting documentation, and that
 * the name of M.I.T. not be used in advertising or publicity pertaining
 * to distribution of the software without specific, written prior
 * permission.  M.I.T. makes no representations about the suitability of
 * this software for any purpose.  It is provided "as is" without express
 * or implied warranty.
 */

/* $Id$ */

#ifdef AUTHENTICATION
Authenticator *findauthenticator (int, int);

int auth_wait (char *, size_t);
void auth_disable_name (char *);
void auth_finished (Authenticator *, int);
void auth_gen_printsub (unsigned char *, size_t, unsigned char *, size_t);
void auth_init (const char *, int);
void auth_is (unsigned char *, int);
void auth_name(unsigned char*, int);
void auth_reply (unsigned char *, int);
void auth_request (void);
void auth_send (unsigned char *, int);
void auth_send_retry (void);
void auth_printsub(unsigned char*, size_t, unsigned char*, size_t);
int getauthmask(char *type, int *maskp);
int auth_enable(char *type);
int auth_disable(char *type);
int auth_onoff(char *type, int on);
int auth_togdebug(int on);
int auth_status(void);
int auth_sendname(unsigned char *cp, int len);
void auth_debug(int mode);

#ifdef UNSAFE
int unsafe_init (Authenticator *, int);
int unsafe_send (Authenticator *);
void unsafe_is (Authenticator *, unsigned char *, int);
void unsafe_reply (Authenticator *, unsigned char *, int);
int unsafe_status (Authenticator *, char *, int);
void unsafe_printsub (unsigned char *, size_t, unsigned char *, size_t);
#endif

#ifdef SRA
int sra_init (Authenticator *, int);
int sra_send (Authenticator *);
void sra_is (Authenticator *, unsigned char *, int);
void sra_reply (Authenticator *, unsigned char *, int);
int sra_status (Authenticator *, char *, int);
void sra_printsub (unsigned char *, size_t, unsigned char *, size_t);
#endif

#ifdef	KRB5
int kerberos5_init (Authenticator *, int);
int kerberos5_send_mutual (Authenticator *);
int kerberos5_send_oneway (Authenticator *);
void kerberos5_is (Authenticator *, unsigned char *, int);
void kerberos5_reply (Authenticator *, unsigned char *, int);
int kerberos5_status (Authenticator *, char *, size_t, int);
void kerberos5_printsub (unsigned char *, size_t, unsigned char *, size_t);
int kerberos5_set_forward(int);
int kerberos5_set_forwardable(int);
#endif
#endif
