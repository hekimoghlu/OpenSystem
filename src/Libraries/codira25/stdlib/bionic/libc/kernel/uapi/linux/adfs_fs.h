/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 2, 2022.
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
#ifndef _UAPI_ADFS_FS_H
#define _UAPI_ADFS_FS_H
#include <linux/types.h>
#include <linux/magic.h>
struct adfs_discrecord {
  __u8 log2secsize;
  __u8 secspertrack;
  __u8 heads;
  __u8 density;
  __u8 idlen;
  __u8 log2bpmb;
  __u8 skew;
  __u8 bootoption;
  __u8 lowsector;
  __u8 nzones;
  __le16 zone_spare;
  __le32 root;
  __le32 disc_size;
  __le16 disc_id;
  __u8 disc_name[10];
  __le32 disc_type;
  __le32 disc_size_high;
  __u8 log2sharesize : 4;
  __u8 unused40 : 4;
  __u8 big_flag : 1;
  __u8 unused41 : 7;
  __u8 nzones_high;
  __u8 reserved43;
  __le32 format_version;
  __le32 root_size;
  __u8 unused52[60 - 52];
} __attribute__((packed, aligned(4)));
#define ADFS_DISCRECORD (0xc00)
#define ADFS_DR_OFFSET (0x1c0)
#define ADFS_DR_SIZE 60
#define ADFS_DR_SIZE_BITS (ADFS_DR_SIZE << 3)
#endif
