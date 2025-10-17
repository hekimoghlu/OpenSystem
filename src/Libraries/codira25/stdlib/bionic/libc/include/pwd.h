/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 1, 2025.
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
/*-
 * Portions Copyright(C) 1995, Jason Downs.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR(S) ``AS IS'' AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR(S) BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

#ifndef _PWD_H_
#define _PWD_H_

#include <sys/cdefs.h>
#include <sys/types.h>

__BEGIN_DECLS

struct passwd {
  char* _Nullable pw_name;
  char* _Nullable pw_passwd;
  uid_t pw_uid;
  gid_t pw_gid;
#ifdef __LP64__
  char* _Nullable pw_gecos;
#else
  /* Note: On LP32, we define pw_gecos to pw_passwd since they're both NULL. */
# define pw_gecos pw_passwd
#endif
  char* _Nullable pw_dir;
  char* _Nullable pw_shell;
};

struct passwd* _Nullable getpwnam(const char* _Nonnull __name);
struct passwd* _Nullable getpwuid(uid_t __uid);

/* Note: Android has thousands and thousands of ids to iterate through */

#if __BIONIC_AVAILABILITY_GUARD(26)
struct passwd* _Nullable getpwent(void) __INTRODUCED_IN(26);

void setpwent(void) __INTRODUCED_IN(26);
void endpwent(void) __INTRODUCED_IN(26);
#endif /* __BIONIC_AVAILABILITY_GUARD(26) */


int getpwnam_r(const char* _Nonnull __name, struct passwd* _Nonnull __pwd, char* _Nonnull __buf, size_t __n, struct passwd* _Nullable * _Nonnull __result);
int getpwuid_r(uid_t __uid, struct passwd* _Nonnull __pwd, char* _Nonnull __buf, size_t __n, struct passwd* _Nullable * _Nonnull __result);

__END_DECLS

#endif
