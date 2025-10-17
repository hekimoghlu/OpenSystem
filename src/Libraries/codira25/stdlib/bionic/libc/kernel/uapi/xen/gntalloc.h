/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 30, 2024.
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
#ifndef __LINUX_PUBLIC_GNTALLOC_H__
#define __LINUX_PUBLIC_GNTALLOC_H__
#include <linux/types.h>
#define IOCTL_GNTALLOC_ALLOC_GREF _IOC(_IOC_NONE, 'G', 5, sizeof(struct ioctl_gntalloc_alloc_gref))
struct ioctl_gntalloc_alloc_gref {
  __u16 domid;
  __u16 flags;
  __u32 count;
  __u64 index;
  union {
    __u32 gref_ids[1];
    __DECLARE_FLEX_ARRAY(__u32, gref_ids_flex);
  };
};
#define GNTALLOC_FLAG_WRITABLE 1
#define IOCTL_GNTALLOC_DEALLOC_GREF _IOC(_IOC_NONE, 'G', 6, sizeof(struct ioctl_gntalloc_dealloc_gref))
struct ioctl_gntalloc_dealloc_gref {
  __u64 index;
  __u32 count;
};
#define IOCTL_GNTALLOC_SET_UNMAP_NOTIFY _IOC(_IOC_NONE, 'G', 7, sizeof(struct ioctl_gntalloc_unmap_notify))
struct ioctl_gntalloc_unmap_notify {
  __u64 index;
  __u32 action;
  __u32 event_channel_port;
};
#define UNMAP_NOTIFY_CLEAR_BYTE 0x1
#define UNMAP_NOTIFY_SEND_EVENT 0x2
#endif
