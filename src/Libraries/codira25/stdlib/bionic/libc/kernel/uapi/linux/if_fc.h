/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 24, 2025.
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
#ifndef _LINUX_IF_FC_H
#define _LINUX_IF_FC_H
#include <linux/types.h>
#define FC_ALEN 6
#define FC_HLEN (sizeof(struct fch_hdr) + sizeof(struct fcllc))
#define FC_ID_LEN 3
#define EXTENDED_SAP 0xAA
#define UI_CMD 0x03
struct fch_hdr {
  __u8 daddr[FC_ALEN];
  __u8 saddr[FC_ALEN];
};
struct fcllc {
  __u8 dsap;
  __u8 ssap;
  __u8 llc;
  __u8 protid[3];
  __be16 ethertype;
};
#endif
