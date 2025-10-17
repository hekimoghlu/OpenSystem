/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 22, 2025.
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
#ifndef _UAPI_LINUX_ASPEED_P2A_CTRL_H
#define _UAPI_LINUX_ASPEED_P2A_CTRL_H
#include <linux/ioctl.h>
#include <linux/types.h>
#define ASPEED_P2A_CTRL_READ_ONLY 0
#define ASPEED_P2A_CTRL_READWRITE 1
struct aspeed_p2a_ctrl_mapping {
  __u64 addr;
  __u32 length;
  __u32 flags;
};
#define __ASPEED_P2A_CTRL_IOCTL_MAGIC 0xb3
#define ASPEED_P2A_CTRL_IOCTL_SET_WINDOW _IOW(__ASPEED_P2A_CTRL_IOCTL_MAGIC, 0x00, struct aspeed_p2a_ctrl_mapping)
#define ASPEED_P2A_CTRL_IOCTL_GET_MEMORY_CONFIG _IOWR(__ASPEED_P2A_CTRL_IOCTL_MAGIC, 0x01, struct aspeed_p2a_ctrl_mapping)
#endif
