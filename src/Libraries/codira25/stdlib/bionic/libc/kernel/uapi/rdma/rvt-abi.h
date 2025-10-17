/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 16, 2025.
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
#ifndef RVT_ABI_USER_H
#define RVT_ABI_USER_H
#include <linux/types.h>
#include <rdma/ib_user_verbs.h>
#ifndef RDMA_ATOMIC_UAPI
#define RDMA_ATOMIC_UAPI(_type,_name) struct { _type val; } _name
#endif
struct rvt_wqe_sge {
  __aligned_u64 addr;
  __u32 length;
  __u32 lkey;
};
struct rvt_cq_wc {
  RDMA_ATOMIC_UAPI(__u32, head);
  RDMA_ATOMIC_UAPI(__u32, tail);
  struct ib_uverbs_wc uqueue[];
};
struct rvt_rwqe {
  __u64 wr_id;
  __u8 num_sge;
  __u8 padding[7];
  struct rvt_wqe_sge sg_list[];
};
struct rvt_rwq {
  RDMA_ATOMIC_UAPI(__u32, head);
  RDMA_ATOMIC_UAPI(__u32, tail);
  struct rvt_rwqe wq[];
};
#endif
