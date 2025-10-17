/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 27, 2021.
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
#ifndef _UAPI_LINUX_UDMABUF_H
#define _UAPI_LINUX_UDMABUF_H
#include <linux/types.h>
#include <linux/ioctl.h>
#define UDMABUF_FLAGS_CLOEXEC 0x01
struct udmabuf_create {
  __u32 memfd;
  __u32 flags;
  __u64 offset;
  __u64 size;
};
struct udmabuf_create_item {
  __u32 memfd;
  __u32 __pad;
  __u64 offset;
  __u64 size;
};
struct udmabuf_create_list {
  __u32 flags;
  __u32 count;
  struct udmabuf_create_item list[];
};
#define UDMABUF_CREATE _IOW('u', 0x42, struct udmabuf_create)
#define UDMABUF_CREATE_LIST _IOW('u', 0x43, struct udmabuf_create_list)
#endif
