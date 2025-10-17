/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 18, 2023.
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
#ifndef _SYS_VFS_H_
#define _SYS_VFS_H_

#include <sys/cdefs.h>

#include <stdint.h>
#include <sys/types.h>

__BEGIN_DECLS

/* The kernel's __kernel_fsid_t has a 'val' member but glibc uses '__val'. */
typedef struct { int __val[2]; } __fsid_t;
typedef __fsid_t fsid_t;

#if defined(__LP64__)
/* We can't just use the kernel struct statfs directly here because
 * it's reused for both struct statfs *and* struct statfs64. */
#define __STATFS64_BODY \
  uint64_t f_type; \
  uint64_t f_bsize; \
  uint64_t f_blocks; \
  uint64_t f_bfree; \
  uint64_t f_bavail; \
  uint64_t f_files; \
  uint64_t f_ffree; \
  fsid_t f_fsid; \
  uint64_t f_namelen; \
  uint64_t f_frsize; \
  uint64_t f_flags; \
  uint64_t f_spare[4]; \

#else
/* 32-bit ARM or x86 (corresponds to the kernel's statfs64 type). */
#define __STATFS64_BODY \
  uint32_t f_type; \
  uint32_t f_bsize; \
  uint64_t f_blocks; \
  uint64_t f_bfree; \
  uint64_t f_bavail; \
  uint64_t f_files; \
  uint64_t f_ffree; \
  fsid_t f_fsid; \
  uint32_t f_namelen; \
  uint32_t f_frsize; \
  uint32_t f_flags; \
  uint32_t f_spare[4]; \

#endif

struct statfs { __STATFS64_BODY };
struct statfs64 { __STATFS64_BODY };

#undef __STATFS64_BODY

/* Declare that we have the f_namelen, f_frsize, and f_flags fields. */
#define _STATFS_F_NAMELEN
#define _STATFS_F_FRSIZE
#define _STATFS_F_FLAGS

/* Pull in the kernel magic numbers. */
#include <linux/magic.h>
/* Add in ones that we had historically that aren't in the uapi header. */
#define BEFS_SUPER_MAGIC      0x42465331
#define BFS_MAGIC             0x1BADFACE
#define CIFS_MAGIC_NUMBER     0xFF534D42
#define COH_SUPER_MAGIC       0x012FF7B7
#define DEVFS_SUPER_MAGIC     0x1373
#define EXT_SUPER_MAGIC       0x137D
#define EXT2_OLD_SUPER_MAGIC  0xEF51
#define HFS_SUPER_MAGIC       0x4244
#define JFS_SUPER_MAGIC       0x3153464a
#define NTFS_SB_MAGIC         0x5346544e
#define ROMFS_MAGIC           0x7275
#define SYSV2_SUPER_MAGIC     0x012FF7B6
#define SYSV4_SUPER_MAGIC     0x012FF7B5
#define UDF_SUPER_MAGIC       0x15013346
#define UFS_MAGIC             0x00011954
#define VXFS_SUPER_MAGIC      0xa501FCF5
#define XENIX_SUPER_MAGIC     0x012FF7B4
#define XFS_SUPER_MAGIC       0x58465342

int statfs(const char* _Nonnull __path, struct statfs* _Nonnull __buf);
int statfs64(const char* _Nonnull __path, struct statfs64* _Nonnull __buf);
int fstatfs(int __fd, struct statfs* _Nonnull __buf);
int fstatfs64(int __fd, struct statfs64* _Nonnull __buf);

__END_DECLS

#endif
