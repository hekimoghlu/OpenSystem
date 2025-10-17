/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 26, 2025.
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
#ifndef _UAPI_GENERIC_STATFS_H
#define _UAPI_GENERIC_STATFS_H
#include <linux/types.h>
#ifndef __statfs_word
#if __BITS_PER_LONG == 64
#define __statfs_word __kernel_long_t
#else
#define __statfs_word __u32
#endif
#endif
struct statfs {
  __statfs_word f_type;
  __statfs_word f_bsize;
  __statfs_word f_blocks;
  __statfs_word f_bfree;
  __statfs_word f_bavail;
  __statfs_word f_files;
  __statfs_word f_ffree;
  __kernel_fsid_t f_fsid;
  __statfs_word f_namelen;
  __statfs_word f_frsize;
  __statfs_word f_flags;
  __statfs_word f_spare[4];
};
#ifndef ARCH_PACK_STATFS64
#define ARCH_PACK_STATFS64
#endif
struct statfs64 {
  __statfs_word f_type;
  __statfs_word f_bsize;
  __u64 f_blocks;
  __u64 f_bfree;
  __u64 f_bavail;
  __u64 f_files;
  __u64 f_ffree;
  __kernel_fsid_t f_fsid;
  __statfs_word f_namelen;
  __statfs_word f_frsize;
  __statfs_word f_flags;
  __statfs_word f_spare[4];
} ARCH_PACK_STATFS64;
#ifndef ARCH_PACK_COMPAT_STATFS64
#define ARCH_PACK_COMPAT_STATFS64
#endif
struct compat_statfs64 {
  __u32 f_type;
  __u32 f_bsize;
  __u64 f_blocks;
  __u64 f_bfree;
  __u64 f_bavail;
  __u64 f_files;
  __u64 f_ffree;
  __kernel_fsid_t f_fsid;
  __u32 f_namelen;
  __u32 f_frsize;
  __u32 f_flags;
  __u32 f_spare[4];
} ARCH_PACK_COMPAT_STATFS64;
#endif
