/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 12, 2024.
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
#include <TargetConditionals.h>

#define INFO_FILE 1
#if !(TARGET_OS_IPHONE && !TARGET_OS_SIMULATOR)
#define INFO_NIS 2
#define INFO_OPEN_DIRECTORY 3
#define INFO_PAM 4
#endif

extern int file_passwd(char *, char *);
#ifdef INFO_NIS
extern int nis_passwd(char *, char *);
#endif
#ifdef INFO_OPEN_DIRECTORY
extern int od_passwd(char *, char *, char*);
#endif
#ifdef INFO_PAM
extern int pam_passwd(char *);
#endif

void
getpasswd(char *name, int isroot, int minlen, int mixcase, int nonalpha,
          char *old_pw, char **new_pw, char **old_clear, char **new_clear);
