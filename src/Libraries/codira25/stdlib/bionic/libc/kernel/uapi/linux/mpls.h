/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 9, 2024.
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
#ifndef _UAPI_MPLS_H
#define _UAPI_MPLS_H
#include <linux/types.h>
#include <asm/byteorder.h>
struct mpls_label {
  __be32 entry;
};
#define MPLS_LS_LABEL_MASK 0xFFFFF000
#define MPLS_LS_LABEL_SHIFT 12
#define MPLS_LS_TC_MASK 0x00000E00
#define MPLS_LS_TC_SHIFT 9
#define MPLS_LS_S_MASK 0x00000100
#define MPLS_LS_S_SHIFT 8
#define MPLS_LS_TTL_MASK 0x000000FF
#define MPLS_LS_TTL_SHIFT 0
#define MPLS_LABEL_IPV4NULL 0
#define MPLS_LABEL_RTALERT 1
#define MPLS_LABEL_IPV6NULL 2
#define MPLS_LABEL_IMPLNULL 3
#define MPLS_LABEL_ENTROPY 7
#define MPLS_LABEL_GAL 13
#define MPLS_LABEL_OAMALERT 14
#define MPLS_LABEL_EXTENSION 15
#define MPLS_LABEL_FIRST_UNRESERVED 16
enum {
  MPLS_STATS_UNSPEC,
  MPLS_STATS_LINK,
  __MPLS_STATS_MAX,
};
#define MPLS_STATS_MAX (__MPLS_STATS_MAX - 1)
struct mpls_link_stats {
  __u64 rx_packets;
  __u64 tx_packets;
  __u64 rx_bytes;
  __u64 tx_bytes;
  __u64 rx_errors;
  __u64 tx_errors;
  __u64 rx_dropped;
  __u64 tx_dropped;
  __u64 rx_noroute;
};
#endif
