/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 30, 2023.
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
#ifndef __LINUX_ROMFS_FS_H
#define __LINUX_ROMFS_FS_H
#include <linux/types.h>
#include <linux/fs.h>
#define ROMBSIZE BLOCK_SIZE
#define ROMBSBITS BLOCK_SIZE_BITS
#define ROMBMASK (ROMBSIZE - 1)
#define ROMFS_MAGIC 0x7275
#define ROMFS_MAXFN 128
#define __mkw(h,l) (((h) & 0x00ff) << 8 | ((l) & 0x00ff))
#define __mkl(h,l) (((h) & 0xffff) << 16 | ((l) & 0xffff))
#define __mk4(a,b,c,d) cpu_to_be32(__mkl(__mkw(a, b), __mkw(c, d)))
#define ROMSB_WORD0 __mk4('-', 'r', 'o', 'm')
#define ROMSB_WORD1 __mk4('1', 'f', 's', '-')
struct romfs_super_block {
  __be32 word0;
  __be32 word1;
  __be32 size;
  __be32 checksum;
  char name[];
};
struct romfs_inode {
  __be32 next;
  __be32 spec;
  __be32 size;
  __be32 checksum;
  char name[];
};
#define ROMFH_TYPE 7
#define ROMFH_HRD 0
#define ROMFH_DIR 1
#define ROMFH_REG 2
#define ROMFH_SYM 3
#define ROMFH_BLK 4
#define ROMFH_CHR 5
#define ROMFH_SCK 6
#define ROMFH_FIF 7
#define ROMFH_EXEC 8
#define ROMFH_SIZE 16
#define ROMFH_PAD (ROMFH_SIZE - 1)
#define ROMFH_MASK (~ROMFH_PAD)
#endif
