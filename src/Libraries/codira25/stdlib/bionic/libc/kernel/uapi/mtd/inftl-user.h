/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 21, 2024.
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
#ifndef __MTD_INFTL_USER_H__
#define __MTD_INFTL_USER_H__
#include <linux/types.h>
#define OSAK_VERSION 0x5120
#define PERCENTUSED 98
#define SECTORSIZE 512
struct inftl_bci {
  __u8 ECCsig[6];
  __u8 Status;
  __u8 Status1;
} __attribute__((packed));
struct inftl_unithead1 {
  __u16 virtualUnitNo;
  __u16 prevUnitNo;
  __u8 ANAC;
  __u8 NACs;
  __u8 parityPerField;
  __u8 discarded;
} __attribute__((packed));
struct inftl_unithead2 {
  __u8 parityPerField;
  __u8 ANAC;
  __u16 prevUnitNo;
  __u16 virtualUnitNo;
  __u8 NACs;
  __u8 discarded;
} __attribute__((packed));
struct inftl_unittail {
  __u8 Reserved[4];
  __u16 EraseMark;
  __u16 EraseMark1;
} __attribute__((packed));
union inftl_uci {
  struct inftl_unithead1 a;
  struct inftl_unithead2 b;
  struct inftl_unittail c;
};
struct inftl_oob {
  struct inftl_bci b;
  union inftl_uci u;
};
struct INFTLPartition {
  __u32 virtualUnits;
  __u32 firstUnit;
  __u32 lastUnit;
  __u32 flags;
  __u32 spareUnits;
  __u32 Reserved0;
  __u32 Reserved1;
} __attribute__((packed));
struct INFTLMediaHeader {
  char bootRecordID[8];
  __u32 NoOfBootImageBlocks;
  __u32 NoOfBinaryPartitions;
  __u32 NoOfBDTLPartitions;
  __u32 BlockMultiplierBits;
  __u32 FormatFlags;
  __u32 OsakVersion;
  __u32 PercentUsed;
  struct INFTLPartition Partitions[4];
} __attribute__((packed));
#define INFTL_BINARY 0x20000000
#define INFTL_BDTL 0x40000000
#define INFTL_LAST 0x80000000
#endif
