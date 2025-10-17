/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 20, 2025.
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
#ifndef _SCSI_SCSI_H
#define _SCSI_SCSI_H
#include <linux/types.h>
#include <asm/param.h>
#include <scsi/scsi_proto.h>
#include <scsi/scsi_status.h>
struct ccs_modesel_head {
  __u8 _r1;
  __u8 medium;
  __u8 _r2;
  __u8 block_desc_length;
  __u8 density;
  __u8 number_blocks_hi;
  __u8 number_blocks_med;
  __u8 number_blocks_lo;
  __u8 _r3;
  __u8 block_length_hi;
  __u8 block_length_med;
  __u8 block_length_lo;
};
#define EXTENDED_MODIFY_DATA_POINTER 0x00
#define EXTENDED_SDTR 0x01
#define EXTENDED_EXTENDED_IDENTIFY 0x02
#define EXTENDED_WDTR 0x03
#define EXTENDED_PPR 0x04
#define EXTENDED_MODIFY_BIDI_DATA_PTR 0x05
#define SCSI_IOCTL_GET_IDLUN 0x5382
#define SCSI_IOCTL_PROBE_HOST 0x5385
#define SCSI_IOCTL_GET_BUS_NUMBER 0x5386
#define SCSI_IOCTL_GET_PCI 0x5387
#endif
