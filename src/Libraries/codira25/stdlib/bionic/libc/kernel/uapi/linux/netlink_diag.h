/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 26, 2025.
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
#ifndef __NETLINK_DIAG_H__
#define __NETLINK_DIAG_H__
#include <linux/types.h>
struct netlink_diag_req {
  __u8 sdiag_family;
  __u8 sdiag_protocol;
  __u16 pad;
  __u32 ndiag_ino;
  __u32 ndiag_show;
  __u32 ndiag_cookie[2];
};
struct netlink_diag_msg {
  __u8 ndiag_family;
  __u8 ndiag_type;
  __u8 ndiag_protocol;
  __u8 ndiag_state;
  __u32 ndiag_portid;
  __u32 ndiag_dst_portid;
  __u32 ndiag_dst_group;
  __u32 ndiag_ino;
  __u32 ndiag_cookie[2];
};
struct netlink_diag_ring {
  __u32 ndr_block_size;
  __u32 ndr_block_nr;
  __u32 ndr_frame_size;
  __u32 ndr_frame_nr;
};
enum {
  NETLINK_DIAG_MEMINFO,
  NETLINK_DIAG_GROUPS,
  NETLINK_DIAG_RX_RING,
  NETLINK_DIAG_TX_RING,
  NETLINK_DIAG_FLAGS,
  __NETLINK_DIAG_MAX,
};
#define NETLINK_DIAG_MAX (__NETLINK_DIAG_MAX - 1)
#define NDIAG_PROTO_ALL ((__u8) ~0)
#define NDIAG_SHOW_MEMINFO 0x00000001
#define NDIAG_SHOW_GROUPS 0x00000002
#define NDIAG_SHOW_RING_CFG 0x00000004
#define NDIAG_SHOW_FLAGS 0x00000008
#define NDIAG_FLAG_CB_RUNNING 0x00000001
#define NDIAG_FLAG_PKTINFO 0x00000002
#define NDIAG_FLAG_BROADCAST_ERROR 0x00000004
#define NDIAG_FLAG_NO_ENOBUFS 0x00000008
#define NDIAG_FLAG_LISTEN_ALL_NSID 0x00000010
#define NDIAG_FLAG_CAP_ACK 0x00000020
#endif
