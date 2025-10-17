/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 30, 2023.
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
#ifndef RDMA_USER_IOCTL_H
#define RDMA_USER_IOCTL_H
#include <rdma/ib_user_mad.h>
#include <rdma/hfi/hfi1_ioctl.h>
#include <rdma/rdma_user_ioctl_cmds.h>
#define IB_IOCTL_MAGIC RDMA_IOCTL_MAGIC
#define IB_USER_MAD_REGISTER_AGENT _IOWR(RDMA_IOCTL_MAGIC, 0x01, struct ib_user_mad_reg_req)
#define IB_USER_MAD_UNREGISTER_AGENT _IOW(RDMA_IOCTL_MAGIC, 0x02, __u32)
#define IB_USER_MAD_ENABLE_PKEY _IO(RDMA_IOCTL_MAGIC, 0x03)
#define IB_USER_MAD_REGISTER_AGENT2 _IOWR(RDMA_IOCTL_MAGIC, 0x04, struct ib_user_mad_reg_req2)
#define HFI1_IOCTL_ASSIGN_CTXT _IOWR(RDMA_IOCTL_MAGIC, 0xE1, struct hfi1_user_info)
#define HFI1_IOCTL_CTXT_INFO _IOW(RDMA_IOCTL_MAGIC, 0xE2, struct hfi1_ctxt_info)
#define HFI1_IOCTL_USER_INFO _IOW(RDMA_IOCTL_MAGIC, 0xE3, struct hfi1_base_info)
#define HFI1_IOCTL_TID_UPDATE _IOWR(RDMA_IOCTL_MAGIC, 0xE4, struct hfi1_tid_info)
#define HFI1_IOCTL_TID_FREE _IOWR(RDMA_IOCTL_MAGIC, 0xE5, struct hfi1_tid_info)
#define HFI1_IOCTL_CREDIT_UPD _IO(RDMA_IOCTL_MAGIC, 0xE6)
#define HFI1_IOCTL_RECV_CTRL _IOW(RDMA_IOCTL_MAGIC, 0xE8, int)
#define HFI1_IOCTL_POLL_TYPE _IOW(RDMA_IOCTL_MAGIC, 0xE9, int)
#define HFI1_IOCTL_ACK_EVENT _IOW(RDMA_IOCTL_MAGIC, 0xEA, unsigned long)
#define HFI1_IOCTL_SET_PKEY _IOW(RDMA_IOCTL_MAGIC, 0xEB, __u16)
#define HFI1_IOCTL_CTXT_RESET _IO(RDMA_IOCTL_MAGIC, 0xEC)
#define HFI1_IOCTL_TID_INVAL_READ _IOWR(RDMA_IOCTL_MAGIC, 0xED, struct hfi1_tid_info)
#define HFI1_IOCTL_GET_VERS _IOR(RDMA_IOCTL_MAGIC, 0xEE, int)
#endif
