/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 13, 2024.
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
#ifndef _UAPI_LINUX_RANDOM_H
#define _UAPI_LINUX_RANDOM_H
#include <linux/types.h>
#include <linux/ioctl.h>
#include <linux/irqnr.h>
#define RNDGETENTCNT _IOR('R', 0x00, int)
#define RNDADDTOENTCNT _IOW('R', 0x01, int)
#define RNDGETPOOL _IOR('R', 0x02, int[2])
#define RNDADDENTROPY _IOW('R', 0x03, int[2])
#define RNDZAPENTCNT _IO('R', 0x04)
#define RNDCLEARPOOL _IO('R', 0x06)
#define RNDRESEEDCRNG _IO('R', 0x07)
struct rand_pool_info {
  int entropy_count;
  int buf_size;
  __u32 buf[];
};
#define GRND_NONBLOCK 0x0001
#define GRND_RANDOM 0x0002
#define GRND_INSECURE 0x0004
struct vgetrandom_opaque_params {
  __u32 size_of_opaque_state;
  __u32 mmap_prot;
  __u32 mmap_flags;
  __u32 reserved[13];
};
#endif
