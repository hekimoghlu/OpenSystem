/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 9, 2024.
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
#ifndef _UAPI_HISI_QM_H
#define _UAPI_HISI_QM_H
#include <linux/types.h>
struct hisi_qp_ctx {
  __u16 id;
  __u16 qc_type;
};
struct hisi_qp_info {
  __u32 sqe_size;
  __u16 sq_depth;
  __u16 cq_depth;
  __u64 reserved;
};
#define HISI_QM_API_VER_BASE "hisi_qm_v1"
#define HISI_QM_API_VER2_BASE "hisi_qm_v2"
#define HISI_QM_API_VER3_BASE "hisi_qm_v3"
#define UACCE_CMD_QM_SET_QP_CTX _IOWR('H', 10, struct hisi_qp_ctx)
#define UACCE_CMD_QM_SET_QP_INFO _IOWR('H', 11, struct hisi_qp_info)
#endif
