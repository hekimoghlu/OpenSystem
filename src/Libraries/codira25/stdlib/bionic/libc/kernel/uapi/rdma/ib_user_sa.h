/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 24, 2022.
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
#ifndef IB_USER_SA_H
#define IB_USER_SA_H
#include <linux/types.h>
enum {
  IB_PATH_GMP = 1,
  IB_PATH_PRIMARY = (1 << 1),
  IB_PATH_ALTERNATE = (1 << 2),
  IB_PATH_OUTBOUND = (1 << 3),
  IB_PATH_INBOUND = (1 << 4),
  IB_PATH_INBOUND_REVERSE = (1 << 5),
  IB_PATH_BIDIRECTIONAL = IB_PATH_OUTBOUND | IB_PATH_INBOUND_REVERSE
};
struct ib_path_rec_data {
  __u32 flags;
  __u32 reserved;
  __u32 path_rec[16];
};
struct ib_user_path_rec {
  __u8 dgid[16];
  __u8 sgid[16];
  __be16 dlid;
  __be16 slid;
  __u32 raw_traffic;
  __be32 flow_label;
  __u32 reversible;
  __u32 mtu;
  __be16 pkey;
  __u8 hop_limit;
  __u8 traffic_class;
  __u8 numb_path;
  __u8 sl;
  __u8 mtu_selector;
  __u8 rate_selector;
  __u8 rate;
  __u8 packet_life_time_selector;
  __u8 packet_life_time;
  __u8 preference;
};
#endif
