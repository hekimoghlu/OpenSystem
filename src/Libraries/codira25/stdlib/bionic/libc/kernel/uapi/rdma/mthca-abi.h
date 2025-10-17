/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 2, 2025.
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
#ifndef MTHCA_ABI_USER_H
#define MTHCA_ABI_USER_H
#include <linux/types.h>
#define MTHCA_UVERBS_ABI_VERSION 1
struct mthca_alloc_ucontext_resp {
  __u32 qp_tab_size;
  __u32 uarc_size;
};
struct mthca_alloc_pd_resp {
  __u32 pdn;
  __u32 reserved;
};
#define MTHCA_MR_DMASYNC 0x1
struct mthca_reg_mr {
  __u32 mr_attrs;
  __u32 reserved;
};
struct mthca_create_cq {
  __u32 lkey;
  __u32 pdn;
  __aligned_u64 arm_db_page;
  __aligned_u64 set_db_page;
  __u32 arm_db_index;
  __u32 set_db_index;
};
struct mthca_create_cq_resp {
  __u32 cqn;
  __u32 reserved;
};
struct mthca_resize_cq {
  __u32 lkey;
  __u32 reserved;
};
struct mthca_create_srq {
  __u32 lkey;
  __u32 db_index;
  __aligned_u64 db_page;
};
struct mthca_create_srq_resp {
  __u32 srqn;
  __u32 reserved;
};
struct mthca_create_qp {
  __u32 lkey;
  __u32 reserved;
  __aligned_u64 sq_db_page;
  __aligned_u64 rq_db_page;
  __u32 sq_db_index;
  __u32 rq_db_index;
};
#endif
