/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 11, 2023.
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
#ifndef _BSD_CYGWIN_UTIL_H
#define _BSD_CYGWIN_UTIL_H

#ifdef HAVE_CYGWIN

#undef ERROR

/* Avoid including windows headers. */
typedef void *HANDLE;
#define INVALID_HANDLE_VALUE ((HANDLE) -1)
#define DNLEN 16
#define UNLEN 256

/* Cygwin functions for which declarations are only available when including
   windows headers, so we have to define them here explicitly. */
extern HANDLE cygwin_logon_user (const struct passwd *, const char *);
extern void cygwin_set_impersonation_token (const HANDLE);

#include <sys/cygwin.h>
#include <io.h>

#define CYGWIN_SSH_PRIVSEP_USER (cygwin_ssh_privsep_user())
const char *cygwin_ssh_privsep_user();

int binary_open(const char *, int , ...);
int check_ntsec(const char *);
char **fetch_windows_environment(void);
void free_windows_environment(char **);
int cygwin_ug_match_pattern_list(const char *, const char *);

#ifndef NO_BINARY_OPEN
#define open binary_open
#endif

#endif /* HAVE_CYGWIN */

#endif /* _BSD_CYGWIN_UTIL_H */
