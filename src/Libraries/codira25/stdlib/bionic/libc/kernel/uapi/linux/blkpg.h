/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 1, 2024.
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
#ifndef _UAPI__LINUX_BLKPG_H
#define _UAPI__LINUX_BLKPG_H
#include <linux/compiler.h>
#include <linux/ioctl.h>
#define BLKPG _IO(0x12, 105)
struct blkpg_ioctl_arg {
  int op;
  int flags;
  int datalen;
  void  * data;
};
#define BLKPG_ADD_PARTITION 1
#define BLKPG_DEL_PARTITION 2
#define BLKPG_RESIZE_PARTITION 3
#define BLKPG_DEVNAMELTH 64
#define BLKPG_VOLNAMELTH 64
struct blkpg_partition {
  long long start;
  long long length;
  int pno;
  char devname[BLKPG_DEVNAMELTH];
  char volname[BLKPG_VOLNAMELTH];
};
#endif
