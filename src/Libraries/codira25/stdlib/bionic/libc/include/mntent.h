/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 13, 2025.
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
#ifndef _MNTENT_H_
#define _MNTENT_H_

#include <sys/cdefs.h>

#include <stdio.h>
#include <paths.h>  /* for _PATH_MOUNTED */

#define MOUNTED _PATH_MOUNTED

#define MNTTYPE_IGNORE "ignore"
#define MNTTYPE_NFS "nfs"
#define MNTTYPE_SWAP "swap"

#define MNTOPT_DEFAULTS "defaults"
#define MNTOPT_NOAUTO "noauto"
#define MNTOPT_NOSUID "nosuid"
#define MNTOPT_RO "ro"
#define MNTOPT_RW "rw"
#define MNTOPT_SUID "suid"

struct mntent {
  char* _Nullable mnt_fsname;
  char* _Nullable mnt_dir;
  char* _Nullable mnt_type;
  char* _Nullable mnt_opts;
  int mnt_freq;
  int mnt_passno;
};

__BEGIN_DECLS

int endmntent(FILE* _Nullable __fp);
struct mntent* _Nullable getmntent(FILE* _Nonnull __fp);
struct mntent* _Nullable getmntent_r(FILE* _Nonnull __fp, struct mntent* _Nonnull __entry, char* _Nonnull __buf, int __size);
FILE* _Nullable setmntent(const char* _Nonnull __filename, const char* _Nonnull __type);

#if __BIONIC_AVAILABILITY_GUARD(26)
char* _Nullable hasmntopt(const struct mntent* _Nonnull __entry, const char* _Nonnull __option) __INTRODUCED_IN(26);
#endif /* __BIONIC_AVAILABILITY_GUARD(26) */


__END_DECLS

#endif
