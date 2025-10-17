/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 16, 2024.
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
#ifndef _LINUX_VIRTIO_MEM_H
#define _LINUX_VIRTIO_MEM_H
#include <linux/types.h>
#include <linux/virtio_types.h>
#include <linux/virtio_ids.h>
#include <linux/virtio_config.h>
#define VIRTIO_MEM_F_ACPI_PXM 0
#define VIRTIO_MEM_F_UNPLUGGED_INACCESSIBLE 1
#define VIRTIO_MEM_F_PERSISTENT_SUSPEND 2
#define VIRTIO_MEM_REQ_PLUG 0
#define VIRTIO_MEM_REQ_UNPLUG 1
#define VIRTIO_MEM_REQ_UNPLUG_ALL 2
#define VIRTIO_MEM_REQ_STATE 3
struct virtio_mem_req_plug {
  __virtio64 addr;
  __virtio16 nb_blocks;
  __virtio16 padding[3];
};
struct virtio_mem_req_unplug {
  __virtio64 addr;
  __virtio16 nb_blocks;
  __virtio16 padding[3];
};
struct virtio_mem_req_state {
  __virtio64 addr;
  __virtio16 nb_blocks;
  __virtio16 padding[3];
};
struct virtio_mem_req {
  __virtio16 type;
  __virtio16 padding[3];
  union {
    struct virtio_mem_req_plug plug;
    struct virtio_mem_req_unplug unplug;
    struct virtio_mem_req_state state;
  } u;
};
#define VIRTIO_MEM_RESP_ACK 0
#define VIRTIO_MEM_RESP_NACK 1
#define VIRTIO_MEM_RESP_BUSY 2
#define VIRTIO_MEM_RESP_ERROR 3
#define VIRTIO_MEM_STATE_PLUGGED 0
#define VIRTIO_MEM_STATE_UNPLUGGED 1
#define VIRTIO_MEM_STATE_MIXED 2
struct virtio_mem_resp_state {
  __virtio16 state;
};
struct virtio_mem_resp {
  __virtio16 type;
  __virtio16 padding[3];
  union {
    struct virtio_mem_resp_state state;
  } u;
};
struct virtio_mem_config {
  __le64 block_size;
  __le16 node_id;
  __u8 padding[6];
  __le64 addr;
  __le64 region_size;
  __le64 usable_region_size;
  __le64 plugged_size;
  __le64 requested_size;
};
#endif
