/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 24, 2022.
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
#include "includes.h"
#ifdef USE_PAM

struct ssh;

void start_pam(struct ssh *);
void finish_pam(void);
u_int do_pam_account(void);
void do_pam_session(struct ssh *);
void do_pam_setcred(void);
void do_pam_chauthtok(void);
int do_pam_putenv(char *, char *);
char ** fetch_pam_environment(void);
char ** fetch_pam_child_environment(void);
void free_pam_environment(char **);
void sshpam_thread_cleanup(void);
void sshpam_cleanup(void);
int sshpam_auth_passwd(Authctxt *, const char *);
int sshpam_get_maxtries_reached(void);
void sshpam_set_maxtries_reached(int);
int is_pam_session_open(void);

#endif /* USE_PAM */
