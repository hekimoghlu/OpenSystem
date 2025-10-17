/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 14, 2022.
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
#ifndef _GRP_H_
#define _GRP_H_

#include <sys/cdefs.h>
#include <sys/types.h>

struct group {
  char* _Nullable gr_name; /* group name */
  char* _Nullable gr_passwd; /* group password */
  gid_t gr_gid; /* group id */
  char* _Nullable * _Nullable gr_mem; /* group members */
};

__BEGIN_DECLS

struct group* _Nullable getgrgid(gid_t __gid);
struct group* _Nullable getgrnam(const char* _Nonnull __name);

/* Note: Android has thousands and thousands of ids to iterate through. */

#if __BIONIC_AVAILABILITY_GUARD(26)
struct group* _Nullable getgrent(void) __INTRODUCED_IN(26);

void setgrent(void) __INTRODUCED_IN(26);
void endgrent(void) __INTRODUCED_IN(26);
#endif /* __BIONIC_AVAILABILITY_GUARD(26) */


#if __BIONIC_AVAILABILITY_GUARD(24)
int getgrgid_r(gid_t __gid, struct group* __BIONIC_COMPLICATED_NULLNESS __group, char* _Nonnull __buf, size_t __n, struct group* _Nullable * _Nonnull __result) __INTRODUCED_IN(24);
int getgrnam_r(const char* _Nonnull __name, struct group* __BIONIC_COMPLICATED_NULLNESS __group, char* _Nonnull __buf, size_t __n, struct group* _Nullable *_Nonnull __result) __INTRODUCED_IN(24);
#endif /* __BIONIC_AVAILABILITY_GUARD(24) */

int getgrouplist(const char* _Nonnull __user, gid_t __group, gid_t* __BIONIC_COMPLICATED_NULLNESS __groups, int* _Nonnull __group_count);
int initgroups(const char* _Nonnull __user, gid_t __group);

__END_DECLS

#endif
