/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 29, 2024.
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
#ifndef __ERDMA_USER_H__
#define __ERDMA_USER_H__
#include <linux/types.h>
#define ERDMA_ABI_VERSION 1
struct erdma_ureq_create_cq {
  __aligned_u64 db_record_va;
  __aligned_u64 qbuf_va;
  __u32 qbuf_len;
  __u32 rsvd0;
};
struct erdma_uresp_create_cq {
  __u32 cq_id;
  __u32 num_cqe;
};
struct erdma_ureq_create_qp {
  __aligned_u64 db_record_va;
  __aligned_u64 qbuf_va;
  __u32 qbuf_len;
  __u32 rsvd0;
};
struct erdma_uresp_create_qp {
  __u32 qp_id;
  __u32 num_sqe;
  __u32 num_rqe;
  __u32 rq_offset;
};
struct erdma_uresp_alloc_ctx {
  __u32 dev_id;
  __u32 pad;
  __u32 sdb_type;
  __u32 sdb_offset;
  __aligned_u64 sdb;
  __aligned_u64 rdb;
  __aligned_u64 cdb;
};
#endif
