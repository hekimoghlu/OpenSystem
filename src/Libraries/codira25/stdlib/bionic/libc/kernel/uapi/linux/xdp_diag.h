/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 15, 2023.
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
#ifndef _LINUX_XDP_DIAG_H
#define _LINUX_XDP_DIAG_H
#include <linux/types.h>
struct xdp_diag_req {
  __u8 sdiag_family;
  __u8 sdiag_protocol;
  __u16 pad;
  __u32 xdiag_ino;
  __u32 xdiag_show;
  __u32 xdiag_cookie[2];
};
struct xdp_diag_msg {
  __u8 xdiag_family;
  __u8 xdiag_type;
  __u16 pad;
  __u32 xdiag_ino;
  __u32 xdiag_cookie[2];
};
#define XDP_SHOW_INFO (1 << 0)
#define XDP_SHOW_RING_CFG (1 << 1)
#define XDP_SHOW_UMEM (1 << 2)
#define XDP_SHOW_MEMINFO (1 << 3)
#define XDP_SHOW_STATS (1 << 4)
enum {
  XDP_DIAG_NONE,
  XDP_DIAG_INFO,
  XDP_DIAG_UID,
  XDP_DIAG_RX_RING,
  XDP_DIAG_TX_RING,
  XDP_DIAG_UMEM,
  XDP_DIAG_UMEM_FILL_RING,
  XDP_DIAG_UMEM_COMPLETION_RING,
  XDP_DIAG_MEMINFO,
  XDP_DIAG_STATS,
  __XDP_DIAG_MAX,
};
#define XDP_DIAG_MAX (__XDP_DIAG_MAX - 1)
struct xdp_diag_info {
  __u32 ifindex;
  __u32 queue_id;
};
struct xdp_diag_ring {
  __u32 entries;
};
#define XDP_DU_F_ZEROCOPY (1 << 0)
struct xdp_diag_umem {
  __u64 size;
  __u32 id;
  __u32 num_pages;
  __u32 chunk_size;
  __u32 headroom;
  __u32 ifindex;
  __u32 queue_id;
  __u32 flags;
  __u32 refs;
};
struct xdp_diag_stats {
  __u64 n_rx_dropped;
  __u64 n_rx_invalid;
  __u64 n_rx_full;
  __u64 n_fill_ring_empty;
  __u64 n_tx_invalid;
  __u64 n_tx_ring_empty;
};
#endif
