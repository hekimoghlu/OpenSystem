/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 3, 2022.
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
#ifndef _UAPI_LINUX_VIRTIO_PMEM_H
#define _UAPI_LINUX_VIRTIO_PMEM_H
#include <linux/types.h>
#include <linux/virtio_ids.h>
#include <linux/virtio_config.h>
#define VIRTIO_PMEM_F_SHMEM_REGION 0
#define VIRTIO_PMEM_SHMEM_REGION_ID 0
struct virtio_pmem_config {
  __le64 start;
  __le64 size;
};
#define VIRTIO_PMEM_REQ_TYPE_FLUSH 0
struct virtio_pmem_resp {
  __le32 ret;
};
struct virtio_pmem_req {
  __le32 type;
};
#endif
