/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 10, 2025.
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
#ifndef _BSD_CRAY_H
#define _BSD_CRAY_H

#ifdef _UNICOS

void cray_init_job(struct passwd *);
void cray_job_termination_handler(int);
void cray_login_failure(char *, int );
int cray_access_denied(char *);
extern char cray_tmpdir[];

#define CUSTOM_FAILED_LOGIN 1

#ifndef IA_SSHD
# define IA_SSHD IA_LOGIN
#endif
#ifndef MAXHOSTNAMELEN
# define MAXHOSTNAMELEN  64
#endif
#ifndef _CRAYT3E
# define TIOCGPGRP (tIOC|20)
#endif

#endif /* UNICOS */

#endif /* _BSD_CRAY_H */
