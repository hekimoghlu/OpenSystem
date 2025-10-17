/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 5, 2023.
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
#ifndef RDMA_USER_IOCTL_CMDS_H
#define RDMA_USER_IOCTL_CMDS_H
#include <linux/types.h>
#include <linux/ioctl.h>
#define RDMA_IOCTL_MAGIC 0x1b
#define RDMA_VERBS_IOCTL _IOWR(RDMA_IOCTL_MAGIC, 1, struct ib_uverbs_ioctl_hdr)
enum {
  UVERBS_ATTR_F_MANDATORY = 1U << 0,
  UVERBS_ATTR_F_VALID_OUTPUT = 1U << 1,
};
struct ib_uverbs_attr {
  __u16 attr_id;
  __u16 len;
  __u16 flags;
  union {
    struct {
      __u8 elem_id;
      __u8 reserved;
    } enum_data;
    __u16 reserved;
  } attr_data;
  union {
    __aligned_u64 data;
    __s64 data_s64;
  };
};
struct ib_uverbs_ioctl_hdr {
  __u16 length;
  __u16 object_id;
  __u16 method_id;
  __u16 num_attrs;
  __aligned_u64 reserved1;
  __u32 driver_id;
  __u32 reserved2;
  struct ib_uverbs_attr attrs[];
};
#endif
