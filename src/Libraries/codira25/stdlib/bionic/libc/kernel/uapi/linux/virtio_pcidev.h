/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 3, 2023.
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
#ifndef _UAPI_LINUX_VIRTIO_PCIDEV_H
#define _UAPI_LINUX_VIRTIO_PCIDEV_H
#include <linux/types.h>
enum virtio_pcidev_ops {
  VIRTIO_PCIDEV_OP_RESERVED = 0,
  VIRTIO_PCIDEV_OP_CFG_READ,
  VIRTIO_PCIDEV_OP_CFG_WRITE,
  VIRTIO_PCIDEV_OP_MMIO_READ,
  VIRTIO_PCIDEV_OP_MMIO_WRITE,
  VIRTIO_PCIDEV_OP_MMIO_MEMSET,
  VIRTIO_PCIDEV_OP_INT,
  VIRTIO_PCIDEV_OP_MSI,
  VIRTIO_PCIDEV_OP_PME,
};
struct virtio_pcidev_msg {
  __u8 op;
  __u8 bar;
  __u16 reserved;
  __u32 size;
  __u64 addr;
  __u8 data[];
};
#endif
