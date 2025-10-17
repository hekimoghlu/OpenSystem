/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 28, 2024.
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
#ifndef _UAPILINUX_SONET_H
#define _UAPILINUX_SONET_H
#define __SONET_ITEMS __HANDLE_ITEM(section_bip); __HANDLE_ITEM(line_bip); __HANDLE_ITEM(path_bip); __HANDLE_ITEM(line_febe); __HANDLE_ITEM(path_febe); __HANDLE_ITEM(corr_hcs); __HANDLE_ITEM(uncorr_hcs); __HANDLE_ITEM(tx_cells); __HANDLE_ITEM(rx_cells);
struct sonet_stats {
#define __HANDLE_ITEM(i) int i
  __SONET_ITEMS
#undef __HANDLE_ITEM
} __attribute__((packed));
#define SONET_GETSTAT _IOR('a', ATMIOC_PHYTYP, struct sonet_stats)
#define SONET_GETSTATZ _IOR('a', ATMIOC_PHYTYP + 1, struct sonet_stats)
#define SONET_SETDIAG _IOWR('a', ATMIOC_PHYTYP + 2, int)
#define SONET_CLRDIAG _IOWR('a', ATMIOC_PHYTYP + 3, int)
#define SONET_GETDIAG _IOR('a', ATMIOC_PHYTYP + 4, int)
#define SONET_SETFRAMING _IOW('a', ATMIOC_PHYTYP + 5, int)
#define SONET_GETFRAMING _IOR('a', ATMIOC_PHYTYP + 6, int)
#define SONET_GETFRSENSE _IOR('a', ATMIOC_PHYTYP + 7, unsigned char[SONET_FRSENSE_SIZE])
#define SONET_INS_SBIP 1
#define SONET_INS_LBIP 2
#define SONET_INS_PBIP 4
#define SONET_INS_FRAME 8
#define SONET_INS_LOS 16
#define SONET_INS_LAIS 32
#define SONET_INS_PAIS 64
#define SONET_INS_HCS 128
#define SONET_FRAME_SONET 0
#define SONET_FRAME_SDH 1
#define SONET_FRSENSE_SIZE 6
#endif
