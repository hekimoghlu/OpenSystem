/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 30, 2023.
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
#ifndef _UAPI_LINUX_VIRTIO_CONFIG_H
#define _UAPI_LINUX_VIRTIO_CONFIG_H
#include <linux/types.h>
#define VIRTIO_CONFIG_S_ACKNOWLEDGE 1
#define VIRTIO_CONFIG_S_DRIVER 2
#define VIRTIO_CONFIG_S_DRIVER_OK 4
#define VIRTIO_CONFIG_S_FEATURES_OK 8
#define VIRTIO_CONFIG_S_NEEDS_RESET 0x40
#define VIRTIO_CONFIG_S_FAILED 0x80
#define VIRTIO_TRANSPORT_F_START 28
#define VIRTIO_TRANSPORT_F_END 42
#ifndef VIRTIO_CONFIG_NO_LEGACY
#define VIRTIO_F_NOTIFY_ON_EMPTY 24
#define VIRTIO_F_ANY_LAYOUT 27
#endif
#define VIRTIO_F_VERSION_1 32
#define VIRTIO_F_ACCESS_PLATFORM 33
#define VIRTIO_F_IOMMU_PLATFORM VIRTIO_F_ACCESS_PLATFORM
#define VIRTIO_F_RING_PACKED 34
#define VIRTIO_F_IN_ORDER 35
#define VIRTIO_F_ORDER_PLATFORM 36
#define VIRTIO_F_SR_IOV 37
#define VIRTIO_F_NOTIFICATION_DATA 38
#define VIRTIO_F_NOTIF_CONFIG_DATA 39
#define VIRTIO_F_RING_RESET 40
#define VIRTIO_F_ADMIN_VQ 41
#endif
