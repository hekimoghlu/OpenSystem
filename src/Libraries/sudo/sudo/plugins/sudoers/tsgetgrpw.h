/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 1, 2024.
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
 * Trivial replacements for the libc getgrent() and getpwent() family
 * of functions for use by testsudoers in the sudo test harness.
 * We need our own since many platforms don't provide set{pw,gr}file().
 */

#include <config.h>

#include <pwd.h>
#include <grp.h>

void testsudoers_setgrfile(const char *);
void testsudoers_setgrent(void);
void testsudoers_endgrent(void);
int testsudoers_setgroupent(int);
struct group *testsudoers_getgrent(void);
struct group *testsudoers_getgrnam(const char *);
struct group *testsudoers_getgrgid(gid_t);

void testsudoers_setpwfile(const char *);
void testsudoers_setpwent(void);
void testsudoers_endpwent(void);
int testsudoers_setpassent(int);
struct passwd *testsudoers_getpwent(void);
struct passwd *testsudoers_getpwnam(const char *);
struct passwd *testsudoers_getpwuid(uid_t);

int testsudoers_getgrouplist2_v1(const char *name, GETGROUPS_T basegid,
    GETGROUPS_T **groupsp, int *ngroupsp);
