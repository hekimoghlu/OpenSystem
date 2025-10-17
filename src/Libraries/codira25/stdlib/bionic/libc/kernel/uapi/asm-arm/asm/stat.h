/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 7, 2023.
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
#ifndef _ASMARM_STAT_H
#define _ASMARM_STAT_H
struct __old_kernel_stat {
  unsigned short st_dev;
  unsigned short st_ino;
  unsigned short st_mode;
  unsigned short st_nlink;
  unsigned short st_uid;
  unsigned short st_gid;
  unsigned short st_rdev;
  unsigned long st_size;
  unsigned long st_atime;
  unsigned long st_mtime;
  unsigned long st_ctime;
};
#define STAT_HAVE_NSEC
struct stat {
  unsigned long st_dev;
  unsigned long st_ino;
  unsigned short st_mode;
  unsigned short st_nlink;
  unsigned short st_uid;
  unsigned short st_gid;
  unsigned long st_rdev;
  unsigned long st_size;
  unsigned long st_blksize;
  unsigned long st_blocks;
  unsigned long st_atime;
  unsigned long st_atime_nsec;
  unsigned long st_mtime;
  unsigned long st_mtime_nsec;
  unsigned long st_ctime;
  unsigned long st_ctime_nsec;
  unsigned long __unused4;
  unsigned long __unused5;
};
struct stat64 {
  unsigned long long st_dev;
  unsigned char __pad0[4];
#define STAT64_HAS_BROKEN_ST_INO 1
  unsigned long __st_ino;
  unsigned int st_mode;
  unsigned int st_nlink;
  unsigned long st_uid;
  unsigned long st_gid;
  unsigned long long st_rdev;
  unsigned char __pad3[4];
  long long st_size;
  unsigned long st_blksize;
  unsigned long long st_blocks;
  unsigned long st_atime;
  unsigned long st_atime_nsec;
  unsigned long st_mtime;
  unsigned long st_mtime_nsec;
  unsigned long st_ctime;
  unsigned long st_ctime_nsec;
  unsigned long long st_ino;
};
#endif
