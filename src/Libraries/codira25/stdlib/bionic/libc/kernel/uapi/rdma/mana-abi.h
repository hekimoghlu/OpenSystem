/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 8, 2023.
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
#ifndef MANA_ABI_USER_H
#define MANA_ABI_USER_H
#include <linux/types.h>
#include <rdma/ib_user_ioctl_verbs.h>
#define MANA_IB_UVERBS_ABI_VERSION 1
enum mana_ib_create_cq_flags {
  MANA_IB_CREATE_RNIC_CQ = 1 << 0,
};
struct mana_ib_create_cq {
  __aligned_u64 buf_addr;
  __u16 flags;
  __u16 reserved0;
  __u32 reserved1;
};
struct mana_ib_create_cq_resp {
  __u32 cqid;
  __u32 reserved;
};
struct mana_ib_create_qp {
  __aligned_u64 sq_buf_addr;
  __u32 sq_buf_size;
  __u32 port;
};
struct mana_ib_create_qp_resp {
  __u32 sqid;
  __u32 cqid;
  __u32 tx_vp_offset;
  __u32 reserved;
};
struct mana_ib_create_rc_qp {
  __aligned_u64 queue_buf[4];
  __u32 queue_size[4];
};
struct mana_ib_create_rc_qp_resp {
  __u32 queue_id[4];
};
struct mana_ib_create_wq {
  __aligned_u64 wq_buf_addr;
  __u32 wq_buf_size;
  __u32 reserved;
};
enum mana_ib_rx_hash_function_flags {
  MANA_IB_RX_HASH_FUNC_TOEPLITZ = 1 << 0,
};
struct mana_ib_create_qp_rss {
  __aligned_u64 rx_hash_fields_mask;
  __u8 rx_hash_function;
  __u8 reserved[7];
  __u32 rx_hash_key_len;
  __u8 rx_hash_key[40];
  __u32 port;
};
struct rss_resp_entry {
  __u32 cqid;
  __u32 wqid;
};
struct mana_ib_create_qp_rss_resp {
  __aligned_u64 num_entries;
  struct rss_resp_entry entries[64];
};
#endif
