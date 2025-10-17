/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 15, 2021.
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
#ifndef LINUX_MMC_IOCTL_H
#define LINUX_MMC_IOCTL_H
#include <linux/types.h>
#include <linux/major.h>
struct mmc_ioc_cmd {
  int write_flag;
  int is_acmd;
  __u32 opcode;
  __u32 arg;
  __u32 response[4];
  unsigned int flags;
  unsigned int blksz;
  unsigned int blocks;
  unsigned int postsleep_min_us;
  unsigned int postsleep_max_us;
  unsigned int data_timeout_ns;
  unsigned int cmd_timeout_ms;
  __u32 __pad;
  __u64 data_ptr;
};
#define mmc_ioc_cmd_set_data(ic,ptr) ic.data_ptr = (__u64) (unsigned long) ptr
struct mmc_ioc_multi_cmd {
  __u64 num_of_cmds;
  struct mmc_ioc_cmd cmds[];
};
#define MMC_IOC_CMD _IOWR(MMC_BLOCK_MAJOR, 0, struct mmc_ioc_cmd)
#define MMC_IOC_MULTI_CMD _IOWR(MMC_BLOCK_MAJOR, 1, struct mmc_ioc_multi_cmd)
#define MMC_IOC_MAX_BYTES (512L * 1024)
#define MMC_IOC_MAX_CMDS 255
#endif
